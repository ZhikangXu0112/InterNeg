import random

def sample_images_per_class(input_txt, output_txt, samples_per_class=16):
    class_to_images = {}
    
    with open(input_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, class_idx = line.rsplit(' ', 1) 
            class_idx = int(class_idx)
            
            if class_idx not in class_to_images:
                class_to_images[class_idx] = []
            class_to_images[class_idx].append(path)

    sampled_lines = []
    for class_idx, images in class_to_images.items():
        if len(images) <= samples_per_class:
            sampled = images  
        else:
            sampled = random.sample(images, samples_per_class)  
        
        for path in sampled:
            sampled_lines.append(f"{path} {class_idx}\n")  
    
    with open(output_txt, 'w') as f:
        f.writelines(sampled_lines)


input_txt = "./benchmark_imglist/imagenet/train_imagenet.txt"     
output_txt = "./benchmark_imglist/imagenet/train_imagenet_16.txt"  
sample_images_per_class(input_txt, output_txt, samples_per_class=16)


