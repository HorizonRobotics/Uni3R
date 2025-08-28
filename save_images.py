import torch
import torchvision.utils as vutils
import os

def save_rendered_images(rendered_images, output_dir="saved_images"):
    """
    保存rendered_images张量中的图片
    
    Args:
        rendered_images: 形状为 [3, 3, 256, 256] 的张量
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保张量在CPU上并且是float类型
    if rendered_images.is_cuda:
        rendered_images = rendered_images.cpu()
    
    # 确保值在[0, 1]范围内
    if rendered_images.max() > 1.0:
        rendered_images = torch.clamp(rendered_images, 0, 1)
    
    # 保存每张图片
    for i in range(rendered_images.shape[0]):
        # 获取第i张图片
        img = rendered_images[i]  # 形状: [3, 256, 256]
        
        # 保存图片
        filename = os.path.join(output_dir, f"rendered_image_{i+1}.png")
        vutils.save_image(img, filename, normalize=False)
        print(f"已保存图片: {filename}")
    
    print(f"所有图片已保存到目录: {output_dir}")

# 在pdb中使用的简化版本
def save_images_simple(rendered_images):
    """
    简化版本，在pdb中直接使用
    """
    # 创建目录
    import os
    os.makedirs("saved_images", exist_ok=True)
    
    # 保存图片
    import torchvision.utils as vutils
    for i in range(rendered_images.shape[0]):
        img = rendered_images[i].cpu()
        filename = f"saved_images/rendered_image_{i+1}.png"
        vutils.save_image(img, filename, normalize=False)
        print(f"已保存: {filename}")
    
    print("所有图片保存完成！")

# 在pdb中直接运行的代码（复制粘贴到pdb中）
pdb_code = """
import os
import torchvision.utils as vutils
os.makedirs("saved_images", exist_ok=True)
for i in range(rendered_images.shape[0]):
    img = rendered_images[i].cpu()
    filename = f"saved_images/rendered_image_{i+1}.png"
    vutils.save_image(img, filename, normalize=False)
    print(f"已保存: {filename}")
print("完成！")
""" 