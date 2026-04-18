"""ComfyUI REST API client with WebSocket progress tracking."""

import io
import json
import uuid
import asyncio
import aiohttp
import aiofiles
import websockets
from pathlib import Path
from typing import Optional, Callable


class ComfyUIClient:
    """Async client for ComfyUI API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    async def is_available(self) -> bool:
        """Check if ComfyUI is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def get_models(self) -> dict:
        """Get available models from ComfyUI."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/object_info") as resp:
                data = await resp.json()
                checkpoints = []
                motion_modules = []
                if "CheckpointLoaderSimple" in data:
                    checkpoints = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
                if "ADE_LoadAnimateDiffModel" in data:
                    motion_modules = data["ADE_LoadAnimateDiffModel"]["input"]["required"]["model_name"][0]
                return {"checkpoints": checkpoints, "motion_modules": motion_modules}

    async def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow prompt and return the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"ComfyUI error {resp.status}: {text}")
                data = await resp.json()
                return data["prompt_id"]

    async def get_queue_status(self) -> dict:
        """Get current queue status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/queue") as resp:
                return await resp.json()

    async def get_history(self, prompt_id: str) -> dict:
        """Get history/output for a completed prompt."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                data = await resp.json()
                return data.get(prompt_id, {})

    async def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an output image/video from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/view", params=params) as resp:
                return await resp.read()

    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 300,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Wait for a prompt to complete via WebSocket, return output info."""
        deadline = asyncio.get_event_loop().time() + timeout
        try:
            async with websockets.connect(f"{self.ws_url}?clientId={self.client_id}") as ws:
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        msg = json.loads(raw)
                        msg_type = msg.get("type")

                        if msg_type == "progress" and progress_callback:
                            data = msg.get("data", {})
                            await progress_callback(data.get("value", 0), data.get("max", 1))

                        elif msg_type == "executing":
                            data = msg.get("data", {})
                            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                                # Generation complete
                                return await self.get_history(prompt_id)

                        elif msg_type == "execution_error":
                            data = msg.get("data", {})
                            raise RuntimeError(f"ComfyUI execution error: {data.get('exception_message', 'Unknown error')}")

                    except asyncio.TimeoutError:
                        # Check if already done
                        history = await self.get_history(prompt_id)
                        if history:
                            return history
        except Exception as e:
            # Fallback: poll history
            pass

        # Polling fallback
        for _ in range(timeout // 5):
            await asyncio.sleep(5)
            history = await self.get_history(prompt_id)
            if history:
                return history

        raise TimeoutError(f"Generation timed out after {timeout}s")

    async def download_outputs(self, history: dict, output_dir: Path, prefix: str = "video") -> list[Path]:
        """Download all video/image outputs from a completed job."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            # Videos
            for video_info in node_output.get("videos", []):
                filename = video_info["filename"]
                subfolder = video_info.get("subfolder", "")
                data = await self.get_image(filename, subfolder, "output")
                out_path = output_dir / f"{prefix}_{filename}"
                async with aiofiles.open(out_path, "wb") as f:
                    await f.write(data)
                saved_paths.append(out_path)

            # Images (for frame output)
            for img_info in node_output.get("images", []):
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                data = await self.get_image(filename, subfolder, "output")
                out_path = output_dir / f"{prefix}_{filename}"
                async with aiofiles.open(out_path, "wb") as f:
                    await f.write(data)
                saved_paths.append(out_path)

        return saved_paths

    async def upload_image(self, image_bytes: bytes, filename: str, overwrite: bool = True) -> str:
        """Upload an image to ComfyUI's input folder.

        Args:
            image_bytes: Raw image data (PNG, JPG, etc.)
            filename:    Desired filename on the server (e.g. "scene_01.png")
            overwrite:   Replace existing file with the same name (default True)

        Returns:
            The server-side filename to use in LoadImage nodes.
        """
        data = aiohttp.FormData()
        data.add_field(
            "image",
            io.BytesIO(image_bytes),
            filename=filename,
            content_type="image/png",
        )
        data.add_field("overwrite", "true" if overwrite else "false")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/upload/image", data=data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Image upload failed {resp.status}: {text}")
                result = await resp.json()
                # ComfyUI returns {"name": "filename.png", "subfolder": "", "type": "input"}
                return result["name"]

    async def get_output_images(self, history: dict) -> list[dict]:
        """Extract image output info from a completed job's history.

        Returns a list of dicts, each with keys:
            filename, subfolder, type  (ready to pass to get_image())
        """
        outputs = []
        for node_id, node_output in history.get("outputs", {}).items():
            for img_info in node_output.get("images", []):
                outputs.append({
                    "filename": img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type":      img_info.get("type", "output"),
                    "node_id":   node_id,
                })
        return outputs

    async def interrupt(self):
        """Interrupt the current generation."""
        async with aiohttp.ClientSession() as session:
            await session.post(f"{self.base_url}/interrupt")
