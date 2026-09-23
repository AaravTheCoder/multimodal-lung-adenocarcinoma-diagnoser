from PIL import Image
import numpy as np

print("🎨 Generating a 224x224 RGB image for model testing...")
# Create a dummy 224x224 RGB array (random textures/noise)
mock_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(mock_array)

# Save it to the exact filename your dataset contract expects
image.save("placeholder_image.png")
print("🎯 Created 'placeholder_image.png' successfully!")