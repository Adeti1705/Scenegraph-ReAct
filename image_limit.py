import sys
import io
import base64
import os
import random

import litellm
from PIL import Image

model = sys.argv[1]
KEY = os.environ["NVIDIA_API_KEY"]

API_BASE = "https://integrate.api.nvidia.com/v1"
IMAGE_COUNTS = [8, 9, 10, 11, 12, 16]
IMAGE_SIZE = (224, 224)


def make_random_image():
    """Generate a unique random RGB image and encode it as a JPEG data URI."""
    width, height = IMAGE_SIZE

    # Random pixel values produce different image content each time.
    pixels = [
        (
            random.randrange(256),
            random.randrange(256),
            random.randrange(256),
        )
        for _ in range(width * height)
    ]

    image = Image.new("RGB", IMAGE_SIZE)
    image.putdata(pixels)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{b64}"
        },
    }


last_success = 0

for n in IMAGE_COUNTS:
    print(f"\nTesting {n} images...", flush=True)

    try:
        # Generate different random image content for every image part.
        image_parts = [make_random_image() for _ in range(n)]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"There are {n} images attached. "
                            "Reply OK if you can process this request."
                        ),
                    },
                    *image_parts,
                ],
            }
        ]

        response = litellm.completion(
            model=model,
            api_base=API_BASE,
            api_key=KEY,
            max_tokens=20,
            num_retries=0,
            max_retries=0,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if answer is not None:
            print(
                f"{n} images: ACCEPTED | Response: {answer!r}",
                flush=True,
            )
            last_success = n
        else:
            print(
                f"{n} images: API returned no text response",
                flush=True,
            )

    except Exception as e:
        print(
            f"{n} images: FAILED | {type(e).__name__}: {e}",
            flush=True,
        )
        break

print(f"\nLargest tested successful count: {last_success}", flush=True)