from PIL import Image, ImageDraw, ImageFont
import os

# Create simple bot icon with different sizes
sizes = [16, 32, 48, 128]
colors = {
    'bg': (102, 126, 234),  # Purple
    'robot': (255, 255, 255)  # White
}

for size in sizes:
    img = Image.new('RGB', (size, size), colors['bg'])
    draw = ImageDraw.Draw(img)
    
    # Draw simple robot face
    # Head
    head_size = int(size * 0.7)
    head_pos = (size - head_size) // 2
    draw.rectangle(
        [head_pos, head_pos, head_pos + head_size, head_pos + head_size],
        outline=colors['robot'],
        width=max(1, size // 32)
    )
    
    # Eyes
    eye_size = max(2, size // 8)
    eye_y = head_pos + head_size // 3
    left_eye_x = head_pos + head_size // 3 - eye_size // 2
    right_eye_x = head_pos + 2 * head_size // 3 - eye_size // 2
    
    draw.ellipse(
        [left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size],
        fill=colors['robot']
    )
    draw.ellipse(
        [right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size],
        fill=colors['robot']
    )
    
    # Mouth
    mouth_y = head_pos + 2 * head_size // 3
    mouth_width = head_size // 2
    mouth_x = (size - mouth_width) // 2
    draw.line(
        [(mouth_x, mouth_y), (mouth_x + mouth_width, mouth_y)],
        fill=colors['robot'],
        width=max(1, size // 32)
    )
    
    # Save
    img.save(f'icon{size}.png')
    print(f'Created icon{size}.png')

print("All icons created successfully!")
