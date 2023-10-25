from PIL import ImageDraw, ImageOps, ImageFilter
import json

from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from .LCM.lcm_pipeline_inpaint import LatentConsistencyModelPipeline_inpaint, LCMScheduler_X
from diffusers import AutoencoderKL, UNet2DConditionModel
#from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
import os
import torch
from PIL import Image
import tomesd

from comfy.cli_args import args


class LCMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device):
        
        save_path = "./lcm_images"
        try:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+"/LCM_Dreamshaper_v7"
        except:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+"\LCM_Dreamshaper_v7"


        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")


        # Initalize Scheduler:
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

        '''
        # Replace the unet with LCM:
        lcm_unet_ckpt = "./Downloads/LCM_Dreamshaper_v7_4k-prune-fp16.safetensors"
        ckpt = load_file(lcm_unet_ckpt)
        m, u = unet.load_state_dict(ckpt, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
        '''

        # LCM Pipeline:
        pipe = LatentConsistencyModelPipeline_inpaint(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        tomesd.apply_patch(pipe, ratio=0.6)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)

class LCMGenerate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Inpaint", "Outpaint"],),	
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "text": ("STRING", {"default": '', "multiline": True}),
                "steps": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),
                
                "width": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }),
                "gfc": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "mask": ("IMAGE", ),
                "original_image": ("IMAGE", ),
                "outpaint_size": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }),
                "outpaint_direction": (["left", "right","top","bottom"],),
                "pipe":("class",)}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,gfc:float,seed: int,image,mask,original_image,outpaint_size,outpaint_direction,mode,pipe):
        '''
        # Save Path:
        save_path = "./lcm_images"

        model_id = folder_paths.get_folder_paths("diffusers")[0]+"/LCM_Dreamshaper_v7"


        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")


        # Initalize Scheduler:
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

        
        # LCM Pipeline:
        pipe = LatentConsistencyModelPipeline_inpaint(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        tomesd.apply_patch(pipe, ratio=0.6)
        #pipe = pipe.to("cuda")
        print("###########: ",type(pipe))'''
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img)).convert("RGB")
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img)).convert("RGB")

        img = original_image[0].numpy()
        img = img*255.0
        original_image = Image.fromarray(np.uint8(img)).convert("RGB")
        
        prompt = text
        seed = seed
        torch.manual_seed(seed)
    # Output Images:

        images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=gfc, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
            
        if mode == "Outpaint":
            if outpaint_direction == "right":
                newbg = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newbg2 = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newmaskbg = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newmaskbg.paste(mask,(outpaint_size,0))
                newbg.paste(original_image,(0,0))
                newbg2.paste(images[0],(outpaint_size,0))
                newmaskbg =newmaskbg.convert('L')
                newmaskbg = ImageOps.invert(newmaskbg)
                image = Image.composite(newbg, newbg2, newmaskbg)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
            elif outpaint_direction == "left":
                newbg = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newbg2 = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newmaskbg = Image.new("RGBA",(images[0].size[0]+outpaint_size,images[0].size[1]),(0,0,0))
                newmaskbg.paste(mask,(0,0))
                newbg.paste(original_image,(outpaint_size,0))
                newbg2.paste(images[0],(0,0))
                newmaskbg =newmaskbg.convert('L')
                newmaskbg = ImageOps.invert(newmaskbg)
                image = Image.composite(newbg, newbg2, newmaskbg)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
            elif outpaint_direction =="top":
                newbg = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newbg2 = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newmaskbg = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newmaskbg.paste(mask,(0,0))
                newbg.paste(original_image,(0,outpaint_size))
                newbg2.paste(images[0],(0,0))
                newmaskbg =newmaskbg.convert('L')
                newmaskbg = ImageOps.invert(newmaskbg)
                image = Image.composite(newbg, newbg2, newmaskbg)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
            elif outpaint_direction == "bottom":
                newbg = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newbg2 = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newmaskbg = Image.new("RGBA",(images[0].size[0],images[0].size[1]+outpaint_size),(0,0,0))
                newmaskbg.paste(mask,(0,outpaint_size))
                newbg.paste(original_image,(0,0))
                newbg2.paste(images[0],(0,outpaint_size))
                newmaskbg =newmaskbg.convert('L')
                newmaskbg = ImageOps.invert(newmaskbg)
                image = Image.composite(newbg, newbg2, newmaskbg)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
        else:
            image = images[0]
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
        return (image,)
        

class LCM_outpaint_prep:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (["left", "right","top","bottom"],),
                "size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }
                ),
                "image": ("IMAGE", ),}
        }
    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE","IMAGE")
    FUNCTION = "outpaint"
    def outpaint(self,image, direction,size):
        growth = 30
        if direction == "right":
            img = image[0].numpy()
            img = img*255.0
            image = Image.fromarray(np.uint8(img)).convert("RGB")
            print(image.size)
            w,h = image.size
            bimage = Image.new("RGB", (w,h), (0,0,0))
            image_crop = image.crop((size,0,image.size[0],image.size[1]))
            bimage.paste(image_crop,(0,0))
            image = bimage
            mask = Image.new("RGB", (w,h), (0,0,0))
            draw = ImageDraw.Draw(mask)
            draw.rectangle((size-growth, 0, image.size[0],image.size[1]), fill=(255,255,255))
            mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
            mask = mask_blur
        elif direction == "left":
            img = image[0].numpy()
            img = img*255.0
            image = Image.fromarray(np.uint8(img)).convert("RGB")
            print(image.size)
            w,h = image.size
            bimage = Image.new("RGB", (w,h), (0,0,0))
            image_crop = image.crop((0,0,image.size[0]-size,image.size[1]))
            bimage.paste(image_crop,(size,0))
            image = bimage
            mask = Image.new("RGB", (w,h), (0,0,0))
            draw = ImageDraw.Draw(mask)
            draw.rectangle((0, 0, image.size[0]-size+growth,image.size[1]), fill=(255,255,255))
            mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
            mask = mask_blur
        elif direction == "top":
            img = image[0].numpy()
            img = img*255.0
            image = Image.fromarray(np.uint8(img)).convert("RGB")
            print(image.size)
            w,h = image.size
            bimage = Image.new("RGB", (w,h), (0,0,0))
            image_crop = image.crop((0,0,image.size[0],image.size[1]-size))
            bimage.paste(image_crop,(0,size))
            image = bimage
            mask = Image.new("RGB", (w,h), (0,0,0))
            draw = ImageDraw.Draw(mask)
            draw.rectangle((0, 0, image.size[0],image.size[1]-size+growth), fill=(255,255,255))
            mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
            mask = mask_blur
        elif direction == "bottom":
            img = image[0].numpy()
            img = img*255.0
            image = Image.fromarray(np.uint8(img)).convert("RGB")
            print(image.size)
            w,h = image.size
            bimage = Image.new("RGB", (w,h), (0,0,0))
            image_crop = image.crop((0,size,image.size[0],image.size[1]))
            bimage.paste(image_crop,(0,0))
            image = bimage
            mask = Image.new("RGB", (w,h), (0,0,0))
            draw = ImageDraw.Draw(mask)
            draw.rectangle((0, image.size[1]-size-growth, image.size[0],image.size[1]), fill=(255,255,255))
            mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
            mask = mask_blur
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
        return (image,mask)





class LoadImageNode_LCM:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": ("STRING", {"multiline": False})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image):
        print(image)
        i = Image.open(image)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

class SaveImage_LCM:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, 512, 512)
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }



NODE_CLASS_MAPPINGS = {
    "LCMGenerate": LCMGenerate,
    "LoadImageNode_LCM":LoadImageNode_LCM,
    "SaveImage_LCM":SaveImage_LCM,
    "LCM_outpaint_prep":LCM_outpaint_prep,
    "LCMLoader":LCMLoader
}
