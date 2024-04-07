from PIL import ImageDraw, ImageOps, ImageFilter
import json
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from .LCM.lcm_pipeline_inpaint import LatentConsistencyModelPipeline_inpaint, LCMScheduler_X
from .LCM.lcm_pipeline_2 import LatentConsistencyModelPipeline_ipadapter
from .LCM.LCM_img2img_pipeline import LatentConsistencyModelPipeline_img2img
from .LCM.LCM_reference_pipeline import LatentConsistencyModelPipeline_reference
from .LCM.LCM_refinpaint_pipeline import LatentConsistencyModelPipeline_refinpaint
from .LCM.pipeline_cn_inpaint import LatentConsistencyModelPipeline_inpaintV2
from .LCM.pipeline_cn_inpaint_ipadapter import LatentConsistencyModelPipeline_inpaintV3
from .LCM.pipeline_inpaint_cn_reference import LatentConsistencyModelPipeline_refinpaintcn
from .LCM.pipeline_cn_reference_img2img import LatentConsistencyModelPipeline_reference_img2img
from .LCM.stable_diffusion_reference_img2img import StableDiffusionImg2ImgPipeline_reference
from .LCM.LCM_lora_inpaint import LCM_inpaint_final
from .LCM.LCM_lora_inpaint_ipadapter import LCM_lora_inpaint_ipadapter
from .LCM.pipeline_cn import LatentConsistencyModelPipeline_controlnet
from .LCM.stable_diffusion_reference_img2img import StableDiffusionImg2ImgPipeline_reference
from .LCM.stable_diffusion_reference_img2img_controlnet import StableDiffusionControlNetImg2ImgPipeline_ref
from .LCM.stable_diffusion_ipadapter_img2img_controlnet import StableDiffusionControlNetImg2ImgPipeline_ipadapter
from diffusers import DDIMScheduler,StableDiffusionControlNetImg2ImgPipeline,StableDiffusionXLPipeline, AutoPipelineForImage2Image,AutoencoderKL, UNet2DConditionModel, T2IAdapter, ControlNetModel, StableDiffusionPipeline, AutoencoderTiny, DiffusionPipeline, LCMScheduler, AutoPipelineForInpainting, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionImg2ImgPipeline
from diffusers.utils import get_class_from_dynamic_module
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
import os
from pathlib import Path
import torch
from PIL import Image
import tomesd
import random
from compel import Compel
import tomesd
from .IPA.ip_adapter import IPAdapter, IPAdapterPlus
from icecream import ic
import utils
import types
from comfy.cli_args import args




#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:192"
def set_max_split_size_mb(model, max_split_size_mb):
    """
    Set the max_split_size_mb parameter in PyTorch to avoid fragmentation.

    Args:
        model (torch.nn.Module): The PyTorch model.
        max_split_size_mb (int): The desired value for max_split_size_mb in megabytes.
    """
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient calculation to prevent unnecessary memory allocations

    # Dummy forward pass to initialize the memory allocator
    dummy_input = torch.randn(1, 1)
    model(dummy_input)

    # Get the current memory allocator state
    allocator = torch.cuda.memory._get_memory_allocator()

    # Update max_split_size_mb in the memory allocator
    allocator.set_max_split_size(max_split_size_mb * 1024 * 1024)



class LCMLoader_controlnet_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "mode":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path,mode):
        
        save_path = "./lcm_images"
        if model_path != "":
            model_id = model_path
        else:
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
        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{mode}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{mode}"
        controlnet = ControlNetModel.from_pretrained(mpath)

        # LCM Pipeline:
        pipe = LatentConsistencyModelPipeline_refinpaintcn(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor,controlnet=controlnet)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)

class LCMLoader_controlnet:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "mode":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path,mode):
        torch.backends.cuda.matmul.allow_tf32 = True
        
        save_path = "./lcm_images"
        if model_path != "":
            model_id = model_path
        else:
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
        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{mode}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{mode}"
        controlnet = ControlNetModel.from_pretrained(mpath)

        # LCM Pipeline:
        pipe = LatentConsistencyModelPipeline_controlnet(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor,controlnet=controlnet)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)


class LCMLoader_img2img:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path):
        
        save_path = "./lcm_images"
        if model_path != "":
            model_id = model_path
        else:
            try:
                model_id = folder_paths.get_folder_paths("diffusers")[0]+"/LCM_Dreamshaper_v7"
            except:
                model_id = folder_paths.get_folder_paths("diffusers")[0]+"\LCM_Dreamshaper_v7"

        
        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",torch_dtype=torch.float32)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder",torch_dtype=torch.float32)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",torch_dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True,torch_dtype=torch.float32)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor",torch_dtype=torch.float32)


        # Initalize Scheduler:
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

        # LCM Pipeline:
        pipe = LatentConsistencyModelPipeline_img2img(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)



class LCMLoader_ReferenceOnly:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "controlnet_model":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path,controlnet_model):
        
        save_path = "./lcm_images"

        if model_path != "":
            model_id = model_path
        else:
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
        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{controlnet_model}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{controlnet_model}"
        controlnet = ControlNetModel.from_pretrained(mpath)

        '''
        # Replace the unet with LCM:
        lcm_unet_ckpt = "./Downloads/LCM_Dreamshaper_v7_4k-prune-fp32.safetensors"
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
        pipe = LatentConsistencyModelPipeline_reference_img2img(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor,controlnet=controlnet)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)


class LCMLoader_RefInpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path):
        
        save_path = "./lcm_images"

        if model_path != "":
            model_id = model_path
        else:
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
        lcm_unet_ckpt = "./Downloads/LCM_Dreamshaper_v7_4k-prune-fp32.safetensors"
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
        pipe = LatentConsistencyModelPipeline_refinpaint(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)


class LCMLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "model_path": ("STRING", {"default": '', "multiline": False}),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,model_path):
        
        save_path = "./lcm_images"

        if model_path != "":
            model_id = model_path
        else:
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
        lcm_unet_ckpt = "./Downloads/LCM_Dreamshaper_v7_4k-prune-fp32.safetensors"
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
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)



class LCMT2IAdapter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "T2Iadapter": ([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)                
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,T2Iadapter):  
        try:
            model_id = folder_paths.get_folder_paths("controlnet")[0]+f"/{T2Iadapter}"
        except:
            model_id = folder_paths.get_folder_paths("controlnet")[0]+f"\{T2Iadapter}"
        self.adapter = adapter = T2IAdapter.from_pretrained(model_id)
        self.adapter = self.adapter.to(torch.device('cuda'))        
        return (adapter,)

class LCM_IPAdapter_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device":(["cpu","cuda"],),
                "ip_adapter":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if str(i).endswith((".ckpt",".safetensors",".bin"))],),
                "ip_adapter_full_path":("STRING", {"default": '', "multiline": False}),
                "mode":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,ip_adapter,ip_adapter_full_path,mode):  
        model_id = folder_paths.get_folder_paths("diffusers")[0]+"/LCM_Dreamshaper_v7"
        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",torch_dtype=torch.float32)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder",torch_dtype=torch.float32)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",torch_dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True,torch_dtype=torch.float32)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor",torch_dtype=torch.float32)


        # Initalize Scheduler:
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

        # LCM Pipeline:
        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{mode}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{mode}"
        controlnet = ControlNetModel.from_pretrained(mpath,torch_dtype=torch.float32)
        pipe = LatentConsistencyModelPipeline_inpaintV3(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor,controlnet=controlnet)
        tomesd.apply_patch(pipe, ratio=0.6)
        '''if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")'''
        if ip_adapter_full_path == "":
            try:
                ip_ckpt = folder_paths.get_folder_paths("controlnet")[0]+f"/{ip_adapter}"
            except:
                ip_ckpt = folder_paths.get_folder_paths("controlnet")[0]+f"\{ip_adapter}"
        else:
            ip_ckpt = ip_adapter_full_path
        image_encoder_path = folder_paths.get_folder_paths("clip_vision")[0]
        
        if "plus" not in ip_adapter:
            ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
        else:
            ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        return (ip_model,)

class LCM_IPAdapter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device":(["cpu","cuda"],),
                "ip_adapter":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if str(i).endswith((".ckpt",".safetensors",".bin"))],),
                "ip_adapter_full_path":("STRING", {"default": '', "multiline": False}),
                
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,ip_adapter,ip_adapter_full_path):  
        model_id = folder_paths.get_folder_paths("diffusers")[0]+"/LCM_Dreamshaper_v7"
        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",torch_dtype=torch.float32)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder",torch_dtype=torch.float32)
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",torch_dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True,torch_dtype=torch.float32)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor",torch_dtype=torch.float32)


        # Initalize Scheduler:
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
  

        # LCM Pipeline:
        
        pipe = LatentConsistencyModelPipeline_ipadapter(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        tomesd.apply_patch(pipe, ratio=0.6)
        '''if device == "cuda":
            pipe.to("cuda")
        else:
            pipe.to("cpu")'''
        if ip_adapter_full_path == "":
            try:
                ip_ckpt = folder_paths.get_folder_paths("controlnet")[0]+f"/{ip_adapter}"
            except:
                ip_ckpt = folder_paths.get_folder_paths("controlnet")[0]+f"\{ip_adapter}"
        else:
            ip_ckpt = ip_adapter_full_path
        image_encoder_path = folder_paths.get_folder_paths("clip_vision")[0]
        
        if "plus" not in ip_adapter:
            ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
        else:
            ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        return (ip_model,)


class LCMGenerate_img2img_IPAdapter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {	
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "loopback":(["disable","enable"],),
                "loopback_iterations":("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                }),
                "image": ("IMAGE", ),
                
                "ip_model":("class",),
                "pil_image":("IMAGE",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,batch,strength,prompt_weighting,loopback,loopback_iterations,pil_image,ip_model,image,scale):
              
        img = pil_image[0].numpy()
        img = img*255.0
        pil_image = Image.fromarray(np.uint8(img))
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:

                images = ip_model.generate(pil_image=pil_image, num_samples=1, num_inference_steps=steps, seed=seed, image=image, strength=strength,scale=scale)
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = ip_model.generate(pil_image=pil_image, num_samples=1, num_inference_steps=steps, seed=seed, image=image, strength=strength,scale=scale)
                
                        res.append(images[0])
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                images = ip_model.generate(pil_image=pil_image, num_samples=1, num_inference_steps=steps, seed=seed, image=image, strength=strength,scale=scale)
                
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = ip_model.generate(pil_image=pil_image, num_samples=1, num_inference_steps=steps, seed=seed, image=image, strength=strength,scale=scale)
                        res.append(images[0])               
            
        return (res,)

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
                "cfg": ("FLOAT", {
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
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "reference_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "Reference_Only":(["disable","enable"],),
                "oupaint_quality":(["higher","lower"],),
                }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,mask,original_image,outpaint_size,outpaint_direction,mode,pipe,batch,prompt_weighting,style_fidelity,reference_image,Reference_Only,oupaint_quality):
        
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img)).convert("RGB")
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img)).convert("RGB")
        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img)).convert("RGB")

        img = original_image[0].numpy()
        img = img*255.0
        original_image = Image.fromarray(np.uint8(img)).convert("RGB")
        res = []
        prompt = text
        if prompt_weighting == "enable":
            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
        
                res.append(images[0])
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
            
        else:
            newres = []
            for i in range(0,1):
                image = res[0]
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                newres.append(image)
            return (res,)
        seed = random.randint(0,1000000000000000)
        torch.manual_seed(seed)
    # Output Images:
        if mode == "Outpaint" and oupaint_quality=="higher":
            newres = []
            if outpaint_direction == "left":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((outpaint_size-30,0,outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "right":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((res[0].size[1]-outpaint_size-30,0,res[0].size[1]-outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "top":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,outpaint_size-30,res[0].size[1]-outpaint_size,outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "bottom":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,res[0].size[1]-outpaint_size-30,res[0].size[0]-outpaint_size,res[0].size[1]-outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            image = image.convert("RGB")
            if Reference_Only == "enable":
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew,ref_image=image,style_fidelity=style_fidelity).images
            else:
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew).images
            newres.append(images[0])
            return (newres,)
        else:
            return ([image],)

class LCMGenerate_inpaintv2:
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
                "cfg": ("FLOAT", {
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
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "reference_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "Reference_Only":(["disable","enable"],),
                "oupaint_quality":(["higher","lower"],),
                "adapter_image": ("IMAGE", ),
                "adapter_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "adapter":("class",),
                "control_image": ("IMAGE", ),
                "control_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,mask,original_image,outpaint_size,outpaint_direction,mode,pipe,batch,prompt_weighting,style_fidelity,reference_image,Reference_Only,oupaint_quality,adapter_image, adapter_weight,adapter,control_image,control_weight):
        
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img)).convert("RGB")
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img)).convert("RGB")
        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img)).convert("RGB")
        img = adapter_image[0].numpy()
        img = img*255.0
        adapter_image = Image.fromarray(np.uint8(img))
        img = control_image[0].numpy()
        img = img*255.0
        control_image = Image.fromarray(np.uint8(img))

        img = original_image[0].numpy()
        img = img*255.0
        original_image = Image.fromarray(np.uint8(img)).convert("RGB")
        res = []
        prompt = text
        if prompt_weighting == "enable":
            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(controlnet_conditioning_scale=control_weight,control_image=control_image,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = pipe(controlnet_conditioning_scale=control_weight,control_image=control_image,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
        
                res.append(images[0])
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
            
        else:
            newres = []
            for i in range(0,1):
                image = res[0]
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                newres.append(image)
            return (res,)
        seed = random.randint(0,1000000000000000)
        torch.manual_seed(seed)
    # Output Images:
        if mode == "Outpaint" and oupaint_quality=="higher":
            newres = []
            if outpaint_direction == "left":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((outpaint_size-30,0,outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "right":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((res[0].size[1]-outpaint_size-30,0,res[0].size[1]-outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "top":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,outpaint_size-30,res[0].size[1]-outpaint_size,outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "bottom":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,res[0].size[1]-outpaint_size-30,res[0].size[0]-outpaint_size,res[0].size[1]-outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            image = image.convert("RGB")
            if Reference_Only == "enable":
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew,ref_image=image,style_fidelity=style_fidelity).images
            else:
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew).images
            newres.append(images[0])
            return (newres,)
        else:
            return ([image],)

class LCMGenerate_inpaintv3:
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
                "cfg": ("FLOAT", {
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
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "reference_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "Reference_Only":(["disable","enable"],),
                "oupaint_quality":(["higher","lower"],),
                "control_image": ("IMAGE", ),
                "control_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "ip_model":("class",),
                "pil_image":("IMAGE",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,ip_model,pil_image,image,scale,mask,original_image,outpaint_size,outpaint_direction,mode,pipe,batch,prompt_weighting,style_fidelity,reference_image,Reference_Only,oupaint_quality,control_image,control_weight):
        
        img = pil_image[0].numpy()
        img = img*255.0
        pil_image = Image.fromarray(np.uint8(img))        
        img = control_image[0].numpy()
        img = img*255.0
        control_image = Image.fromarray(np.uint8(img))
        reference_image = pil_image
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))  
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img))        

        

        img = original_image[0].numpy()
        img = img*255.0
        original_image = Image.fromarray(np.uint8(img))
        res = []
        prompt = text
        if prompt_weighting == "enable":
            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask).images
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                if Reference_Only == "enable":
                    images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = 1.0, image=image, mask_image=mask,ref_image=reference_image,style_fidelity=style_fidelity).images
                else:
                    images = ip_model.generate(pil_image=pil_image, num_samples=1, num_inference_steps=steps, seed=seed, image=image, strength=1.0,scale=scale,mask_image=mask,control_image=control_image)
        
                res.append(images[0])
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
            
        else:
            newres = []
            for i in range(0,1):
                image = res[0]
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                newres.append(image)
            return (res,)
        seed = random.randint(0,1000000000000000)
        torch.manual_seed(seed)
    # Output Images:
        if mode == "Outpaint" and oupaint_quality=="higher":
            newres = []
            if outpaint_direction == "left":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((outpaint_size-30,0,outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "right":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((res[0].size[1]-outpaint_size-30,0,res[0].size[1]-outpaint_size+30,res[0].size[1]-outpaint_size), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "top":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,outpaint_size-30,res[0].size[1]-outpaint_size,outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            elif outpaint_direction == "bottom":
                mask = Image.new("RGB", (image.size[0],image.size[1]), (0,0,0))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((0,res[0].size[1]-outpaint_size-30,res[0].size[0]-outpaint_size,res[0].size[1]-outpaint_size+30), fill=(255,255,255))
                mask_blur = mask.filter(ImageFilter.GaussianBlur(10))
                masknew = mask_blur
            image = image.convert("RGB")
            if Reference_Only == "enable":
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew,ref_image=image,style_fidelity=style_fidelity).images
            else:
                images = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=image.size[0],height=image.size[1],strength = 1.0, image=image, mask_image=masknew).images
            newres.append(images[0])
            return (newres,)
        else:
            return ([image],)

class LCMGenerate_img2img:
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "outpaint_size": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }),
                "outpaint_direction": (["left", "right","top","bottom"],),
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "loopback":(["disable","enable"],),
                "loopback_iterations":("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                }),
                "adapter_image": ("IMAGE", ),
                "adapter_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "adapter":("class",)}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,outpaint_size,outpaint_direction,mode,pipe,batch,strength,prompt_weighting,loopback,loopback_iterations,adapter_image, adapter_weight,adapter):
        
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        img = adapter_image[0].numpy()
        img = img*255.0
        adapter_image = Image.fromarray(np.uint8(img))
        
 
               
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:

                images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image).images
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=images[0]).images
                
                        res.append(images[0])
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image).images
                
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=images[0]).images
                        res.append(images[0])
                
            
        return (res,)


class LCMGenerate_img2img_controlnet:
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "outpaint_size": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 5000,
                    "step": 64,
                }),
                "outpaint_direction": (["left", "right","top","bottom"],),
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "loopback":(["disable","enable"],),
                "loopback_iterations":("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                }),
                "control_image": ("IMAGE", ),
                "control_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,outpaint_size,outpaint_direction,mode,pipe,batch,strength,prompt_weighting,loopback,loopback_iterations, control_image, control_weight):
        
        try:
            img = image[0].numpy()
            img = img*255.0
            image = Image.fromarray(np.uint8(img))
            img = control_image[0].numpy()
            img = img*255.0
            control_image = Image.fromarray(np.uint8(img))
        except:
            image=image
        
        
        
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:

                images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image).images
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=images[0]).images
                
                        res.append(images[0])
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                images = pipe(controlnet_conditioning_scale=control_weight,control_image=control_image,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image).images
                
                res.append(images[0])
                if loopback == "enable" and batch==1:
                    for j in range(0,loopback_iterations):
                        images = pipe(adapter_weight=adapter_weight,adapter_img=adapter_image,adapter=adapter,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=images[0]).images
                        res.append(images[0])
                
        
            
        return (res,)




class LCMGenerate_ReferenceOnly:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "prompt_weighting":(["disable","enable"],),
                "control_image": ("IMAGE", ),
                "control_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                })
                }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,strength,reference_image,pipe,batch,prompt_weighting,style_fidelity,control_image, control_weight):
        
        
      
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img))
        img = control_image[0].numpy()
        img = img*255.0
        control_image = Image.fromarray(np.uint8(img))

        

        
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:

                images = pipe(prompt_embeds=prompt_embeds, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image,ref_image=reference_image,style_fidelity=style_fidelity).images
                res.append(images[0])

        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
            # Output Images:
                images = pipe(controlnet_conditioning_scale=control_weight,control_image=control_image,prompt=prompt, num_images_per_prompt=1, num_inference_steps=steps, guidance_scale=cfg, lcm_origin_steps=50,width=width,height=height,strength = strength, image=image,ref_image=reference_image,style_fidelity=style_fidelity).images
                
                res.append(images[0])

            
            
        return (res,)




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


class FreeU_LCM:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"b1": ("FLOAT", {
                    "default": 1.3,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    }),
                    "b2": ("FLOAT", {
                    "default": 1.4,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    }),
                    "s1": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    }),
                    "s2": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    }),
                    "pipe":("class",)},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("class",)
    FUNCTION = "load_image"
    def load_image(self, b1,b2,s1,s2,pipe):
        pipe.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)
        return (pipe,)

class ImageShuffle:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image_1": ("IMAGE",),
                    "image_2": ("IMAGE",),
                    "image_3": ("IMAGE",),
                    "image_4": ("IMAGE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image_1,image_2,image_3,image_4,seed):
        newarr = [image_1,image_2,image_3,image_4]
        random.shuffle(newarr)
        return (newarr[0],newarr[1],newarr[2],newarr[3])

class ImageOutputToComfyNodes:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": ("IMAGE",)},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image):
        image = np.array(image[0]).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


class ComfyNodesToSaveCanvas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",)},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image):
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        res = [image]
        return (res,)

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
            img = image
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            data = {
                "lastimage":str(os.path.join(full_output_folder, file))
            }
            json_object = json.dumps(data, indent=4)
            savepath = folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasTool/lastimage.json"
            savepath = Path(savepath)
            with open(savepath, "w") as outfile:
                print(savepath)
                outfile.write(json_object)
                
            
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            
        return { "ui": { "images": results } }


class OutpaintCanvasTool:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),                      
                    }
                }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    FUNCTION = "canvasopen"
    def canvasopen(self,seed):
        req = request.Request("http://localhost:5000/getSharedData")
        req.add_header("Content-Type","application/json")
        response = request.urlopen(req).read().decode('utf-8')
        savedata = json.loads(response)["savedata"]
        imgs = json.loads(response)["imgs"]
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/image.png")
        bg = Image.fromarray(np.array(json.loads(imgs["image"]),dtype="uint8"))
        i = ImageOps.exif_transpose(bg)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        bg = torch.from_numpy(image)[None,]
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/mask.png")
        bg2 = Image.fromarray(np.array(json.loads(imgs["mask"]),dtype="uint8"))
        i = ImageOps.exif_transpose(bg2)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        bg2 = torch.from_numpy(image)[None,]
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/reference.png")
        ref = Image.fromarray(np.array(json.loads(imgs["reference"]),dtype="uint8"))
        i = ImageOps.exif_transpose(ref)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        ref = torch.from_numpy(image)[None,]
        
        return (bg,bg2,ref)

class stitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                        "image":("IMAGE",),
                                            
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canvasopen"
    def canvasopen(self,image):
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img)).convert("RGBA")
        '''path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/data.json")
        with open(path,"r") as json_file:
            savedata = json.load(json_file)["savedata"]'''
        req = request.Request("http://localhost:5000/getSharedData")
        req.add_header("Content-Type","application/json")
        response = request.urlopen(req).read().decode('utf-8')
        savedata = json.loads(response)["savedata"]
        imgs = json.loads(response)["imgs"]
        #img = Image.fromarray(np.array(json.loads(imgs["out"]),dtype="uint8"))
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/out.png")
        bg = Image.fromarray(np.array(json.loads(imgs["out"]),dtype="uint8")).convert("RGBA")
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/mask.png")
        msksmall = Image.fromarray(np.array(json.loads(imgs["mask"]),dtype="uint8")).convert("L")
        #path = Path(folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/image.png")
        cropped = Image.fromarray(np.array(json.loads(imgs["image"]),dtype="uint8")).convert("RGBA")
        width = int(savedata["additionaldims"]["right"]) + int(savedata["additionaldims"]["left"]) + bg.size[0]
        height = int(savedata["additionaldims"]["top"]) + int(savedata["additionaldims"]["bottom"]) + bg.size[1]
        new = Image.new("RGBA",(width,height),(0,0,0,0))
        lft = 0
        tp = 0
        image = Image.composite(image, cropped, msksmall)
        if savedata["additionaldims"]["left"]>0:
            lft = int(savedata["additionaldims"]["left"])

        if savedata["additionaldims"]["top"]>0:
            tp = int(savedata["additionaldims"]["top"])
        new.paste(bg,(lft,tp))
        lft = int(savedata["crpdims"]["left"])
        tp = int(savedata["crpdims"]["top"])
        new.paste(image,(lft,tp))
        res = []
        res.append(new)
        return (res,)
        
class LCMLoraLoader_inpaint:
    def __init__(self):
        pass

    

    @classmethod
    def INPUT_TYPES(s):
        files = []
        for j in ["/IPAdapter/models","\IPAdapter\models"]:
            try:
                for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]+j):
                    if os.path.isfile(os.path.join(folder_paths.get_folder_paths("controlnet")[0]+j,i)):
                        files.append(i)
            except:
                pass
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "ip_adapter":(["disable","enable"],),
                "reference_only":(["disable","enable"],),
                "ip_adapter_model":(files,),
                "model_name":([i for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"\{i}")],),
                "controlnet_model":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],)
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,controlnet_model,ip_adapter_model,model_name,ip_adapter,reference_only):        
        try:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"/{model_name}"
        except:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"\{model_name}"       

        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{controlnet_model}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{controlnet_model}"
        controlnet = ControlNetModel.from_pretrained(mpath)
        try:
            path = Path(folder_paths.get_folder_paths("vae")[0]+"/taesd")
            vae2 = AutoencoderTiny.from_pretrained(path, torch_device='cuda', torch_dtype=torch.float32)
        except:
            vae2 = None
        if reference_only == "disable":
            pipe = LCM_lora_inpaint_ipadapter.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2 = vae2)
        else:
            pipe = LCM_inpaint_final.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2 = vae2)
        if ip_adapter=="enable":
            path = Path(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter", subfolder="models", weight_name=ip_adapter_model)
            pipe.load_ip_adapter(path, subfolder="models", weight_name=ip_adapter_model)
        
        pipe.vae2 = pipe.vae2.cuda()
        # set scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # load LCM-LoRA
        path = Path(folder_paths.get_folder_paths("loras")[0]+"/pytorch_lora_weights.safetensors")
        pipe.load_lora_weights(path)
        pipe.fuse_lora()
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            #pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)
class LCMLora_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
                "cfg": ("FLOAT", {
                    "default": 1.8,
                    "min": 0,
                    "max": 3.0,
                    "step": 0.1,
                }),
                "mask": ("IMAGE", ),
                "image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "reference_style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "controlnet_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
                "ipadapter_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width:int,height:int,cfg:float,seed: int,image,pipe,ipadapter_scale,batch,reference_image,reference_style_fidelity,strength,prompt_weighting, controlnet_weight,mask,reference_only,ip_adapter):
        
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img)).convert("RGB")

        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img)).convert("RGB")
        
        if ip_adapter == "enable":
            ip_adapter_image = reference_image            
            pipe.set_ip_adapter_scale(ipadapter_scale)
        else:
            ip_adapter_image = None
 
               
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
                if reference_only == "enable":
                    images = pipe(
                        prompt_embeds=prompt_embeds, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=adapter_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,ref_image=reference_image,style_fidelity=reference_style_fidelity,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                else:
                    images = pipe(
                        prompt_embeds=prompt_embeds, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=adapter_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                res.append(images[0])
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                generator = torch.manual_seed(seed)
                if reference_only == "enable":
                    images = pipe(prompt=prompt, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=controlnet_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,ref_image=reference_image,style_fidelity=reference_style_fidelity,cross_attention_kwargs={"scale": 1}).images
                else:
                    images = pipe(
                        prompt=prompt, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=controlnet_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                res.append(images[0])                
            
        return (res,)
    

class ImageDims:
    def __init__(self):
        pass    

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            }
        }
    RETURN_TYPES = ("NUMBER","NUMBER",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,image):        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))

        return (image.size[0],image.size[1])

class LCMLora_inpaintV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "text": ("STRING", {"default": '', "multiline": True}),
                "steps": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),                
                "width": ("NUMBER",),
                "height": ("NUMBER",),
                "cfg": ("FLOAT", {
                    "default": 1.8,
                    "min": 0,
                    "max": 3.0,
                    "step": 0.1,
                }),
                "mask": ("IMAGE", ),
                "image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "reference_style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "pipe":("class",),
                "batch": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "prompt_weighting":(["disable","enable"],),
                "controlnet_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
                "ipadapter_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, text: str,steps: int,width,height,cfg:float,seed: int,image,pipe,ipadapter_scale,batch,reference_image,reference_style_fidelity,strength,prompt_weighting, controlnet_weight,mask,reference_only,ip_adapter):
        width = int(width)
        height = int(height)
        
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))
        
        img = mask[0].numpy()
        img = img*255.0
        mask = Image.fromarray(np.uint8(img)).convert("RGB")

        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img)).convert("RGB")
        
        if ip_adapter == "enable":
            ip_adapter_image = reference_image            
            pipe.set_ip_adapter_scale(ipadapter_scale)
        else:
            ip_adapter_image = None
 
               
        res = []
        prompt = text
        if prompt_weighting == "enable":

            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
            prompt_embeds = compel_proc(prompt)
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                torch.manual_seed(seed)
                if reference_only == "enable":
                    images = pipe(
                        prompt_embeds=prompt_embeds, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=adapter_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,ref_image=reference_image,style_fidelity=reference_style_fidelity,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                else:
                    images = pipe(
                        prompt_embeds=prompt_embeds, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=adapter_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                res.append(images[0])
        else:
            for i in range(0,batch):
                seed = random.randint(0,1000000000000000)
                generator = torch.manual_seed(seed)
                if reference_only == "enable":
                    images = pipe(prompt=prompt, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=controlnet_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,ref_image=reference_image,style_fidelity=reference_style_fidelity,cross_attention_kwargs={"scale": 1}).images
                else:
                    images = pipe(
                        prompt=prompt, num_inference_steps=steps, generator=generator, guidance_scale=cfg,width=width,height=height,image=image,controlnet_conditioning_scale=controlnet_weight,mask_image=mask,control_image=None,strength=strength,ip_adapter_image= ip_adapter_image,
                        cross_attention_kwargs={"scale": 1}
                    ).images
                res.append(images[0])                
            
        return (res,)
        
class LCMLoader_SDTurbo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ([i for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"\{i}")],),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "reference_only":(["disable","enable"],),
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,tomesd_value,model_path,reference_only):
        
        
        
        try:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f'/{model_path}'
        except:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f'\{model_path}'
       
        pipe = StableDiffusionImg2ImgPipeline_reference.from_pretrained(model_id,safety_checker=None)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_sequential_cpu_offload()
        return (pipe,)

class LCMGenerate_SDTurbo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe":("class",),
                "device": (["cuda", "cpu"],),	
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "negative_prompt": ("STRING", {"default": '', "multiline": True}),
                "prompt": ("STRING", {"default": '', "multiline": True}),
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "batch": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                }),
                "reference_only":(["disable","enable"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, prompt: str,steps: int,width:int,height:int,cfg:float,seed: int,image,pipe,batch,strength,negative_prompt,device,reference_image,style_fidelity,reference_only):
        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))

        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img))
            
        negative_prompt=negative_prompt
        
        res = []
        for m in range(batch):
            if reference_only == "disable":
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=0.0).images      
            else:
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=0.0,ref_image = reference_image,style_fidelity=style_fidelity).images      
            res.append(images[0])                
            
        return (res,)

class Loader_SegmindVega:
    def __init__(self):
        pass

    

    @classmethod
    def INPUT_TYPES(s):
        files = []
        for j in ["/IPAdapter/models","\IPAdapter\models"]:
            try:
                for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]+j):
                    if os.path.isfile(os.path.join(folder_paths.get_folder_paths("controlnet")[0]+j,i)):
                        files.append(i)
            except:
                pass
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "ip_adapter_model":(files,),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
                "model_name":([i for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"\{i}")],),
                
            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,ip_adapter_model,model_name,reference_only,ip_adapter):        
        try:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"/{model_name}"
        except:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"\{model_name}"       
        '''if ip_adapter == "enable" and reference_only=="disable":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None)
            pipe.load_ip_adapter(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter", subfolder="models", weight_name=ip_adapter_model)
        elif reference_only == "disable":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None)
        else:
            pipe = StableDiffusionImg2ImgPipeline_reference.from_pretrained(model_id,safety_checker=None)'''
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id,safety_checker=None)
        # set scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        pipe.load_lora_weights(folder_paths.get_folder_paths("loras")[0]+"/pytorch_lora_weights_vega.safetensors")
        pipe.fuse_lora()
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()
        else:
            pipe.to("cpu")
        return (pipe,)
        
class SegmindVega:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe":("class",),
                "device": (["cuda", "cpu"],),
                "mode": (["variation", "editing"],),	
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "original_prompt": ("STRING", {"default": '', "multiline": True}),
                "prompt": ("STRING", {"default": '', "multiline": True}),
                "negative_prompt": ("STRING", {"default": '', "multiline": True}),
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "editing_early_steps": ("INT", {
                    "default": 1000,
                    "min": 0,
                    "max": 5000,
                    "step": 1,
                }),
                "batch": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                }),
                "ipadapter_scale":("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, prompt: str,steps: int,width:int,height:int,cfg:float,seed: int,image,mode,pipe,batch,strength,editing_early_steps,original_prompt,device,reference_image,style_fidelity,ipadapter_scale,reference_only,ip_adapter,negative_prompt):
        if ip_adapter =="enable":
            pipe.set_ip_adapter_scale(ipadapter_scale)

        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))

        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img))

        
              
        res = []
        for m in range(batch):
            '''if ip_adapter == "enable" and reference_only=="disable":
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = reference_image).images
            elif reference_only == "disable":
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg).images      
            elif reference_only=="enable":
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ref_image=reference_image,style_fidelity=style_fidelity).images      
            else:
                images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = reference_image,ref_image=reference_image,style_fidelity=style_fidelity).images      
            '''
            images = pipe(prompt=prompt, negative_prompt=negative_prompt,num_inference_steps=steps,guidance_scale=cfg,output_type="latent").images
            '''res = []
            for idx, image in enumerate(images):
                res.append(image) '''
            out = {"samples":(images/0.13025)}

            
        return (out,)


class SaveImage_Puzzle:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
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
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            data = {
                "lastimage":str(os.path.join(full_output_folder, file)),
                "done":"y"
            }
            json_object = json.dumps(data, indent=4)
            savepath = folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/puzzle/lastimage.json"
            savepath = Path(savepath)
            with open(savepath, "w") as outfile:
                print(savepath)
                outfile.write(json_object)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class SaveImage_PuzzleV2:
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
            img = image
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            data = {
                "lastimage":str(os.path.join(full_output_folder, file)),
                "done":"y"
            }
            json_object = json.dumps(data, indent=4)
            savepath = folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/puzzle/lastimage.json"
            savepath = Path(savepath)
            with open(savepath, "w") as outfile:
                print(savepath)
                outfile.write(json_object)
            
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            
        return { "ui": { "images": results } }

class LCMLoraLoader_ipadapter:
    def __init__(self):
        pass

    

    @classmethod
    def INPUT_TYPES(s):
        files = []
        for j in ["/IPAdapter/models","\IPAdapter\models"]:
            try:
                for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]+j):
                    if os.path.isfile(os.path.join(folder_paths.get_folder_paths("controlnet")[0]+j,i)):
                        files.append(i)
            except:
                pass
        return {
            "required": {
                "device": (["GPU", "CPU"],),
                "tomesd_value": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "ip_adapter_model":(files,),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
                "control_net":(["disable","enable"],),
                "model_name":([i for i in os.listdir(folder_paths.get_folder_paths("diffusers")[0]) if os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("diffusers")[0]+f"\{i}")],),
                "controlnet_model":([i for i in os.listdir(folder_paths.get_folder_paths("controlnet")[0]) if os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"/{i}") or  os.path.isdir(folder_paths.get_folder_paths("controlnet")[0]+f"\{i}")],),
                "LCM_enable":(["disable","enable"],),

            }
        }
    RETURN_TYPES = ("class",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self,device,tomesd_value,ip_adapter_model,model_name,reference_only,ip_adapter,controlnet_model,control_net,LCM_enable):        
        try:          
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"/{model_name}"
        except:
            model_id = folder_paths.get_folder_paths("diffusers")[0]+f"\{model_name}"       
        try:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"/{controlnet_model}"
        except:
            mpath = folder_paths.get_folder_paths("controlnet")[0]+f"\{controlnet_model}"
        controlnet = ControlNetModel.from_pretrained(mpath)
        try:
            path = Path(folder_paths.get_folder_paths("vae")[0]+"/taesd")
            vae2 = AutoencoderTiny.from_pretrained(path, torch_device='cuda', torch_dtype=torch.float32)
        except:
            vae2 = None
        if control_net == "disable":
            if ip_adapter == "enable" and reference_only=="disable":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None)
                pipe.load_ip_adapter(Path(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter"), subfolder="models", weight_name=ip_adapter_model)
            elif reference_only == "disable":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,safety_checker=None)
            elif reference_only == "enable":
                pipe = StableDiffusionImg2ImgPipeline_reference.from_pretrained(model_id,safety_checker=None)
            else:
                pipe = StableDiffusionImg2ImgPipeline_reference.from_pretrained(model_id,safety_checker=None)
                pipe.load_ip_adapter(Path(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter"), subfolder="models", weight_name=ip_adapter_model)
        else:
            if ip_adapter == "enable" and reference_only=="disable":
                #pipe = StableDiffusionControlNetImg2ImgPipeline_ipadapter.from_pretrained(model_id,safety_checker=None,controlnet=controlnet)
                pipe = StableDiffusionControlNetImg2ImgPipeline_ref.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2=vae2)
                pipe.load_ip_adapter(Path(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter"), subfolder="models", weight_name=ip_adapter_model)
            elif reference_only == "disable":
                #pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_id,safety_checker=None,controlnet=controlnet)
                pipe = StableDiffusionControlNetImg2ImgPipeline_ref.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2=vae2)
            elif ip_adapter == "disable" and reference_only == "enable":
                #pipe = StableDiffusionControlNetImg2ImgPipeline_ref.from_pretrained(model_id,safety_checker=None, controlnet=controlnet,vae2=vae2)
                pipe = StableDiffusionControlNetImg2ImgPipeline_ref.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2=vae2)
            else:
                pipe = StableDiffusionControlNetImg2ImgPipeline_ref.from_pretrained(model_id,safety_checker=None,controlnet=controlnet,vae2=vae2)
                pipe.load_ip_adapter(Path(folder_paths.get_folder_paths("controlnet")[0]+"/IPAdapter"), subfolder="models", weight_name=ip_adapter_model)
        
        
        if LCM_enable == "enable":
            # set scheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # load LCM-LoRA
            try:
                pipe.load_lora_weights(folder_paths.get_folder_paths("loras")[0]+"/pytorch_lora_weights.safetensors")
            except:
                pipe.load_lora_weights(folder_paths.get_folder_paths("loras")[0]+"\pytorch_lora_weights.safetensors")
            pipe.fuse_lora()
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        tomesd.apply_patch(pipe, ratio=tomesd_value)
        if device == "GPU":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cpu")
        return (pipe,)
        
class LCMLora_ipadapter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe":("class",),	
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": '', "multiline": True}),
                "negative_prompt": ("STRING", {"default": '', "multiline": True}),
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
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "image": ("IMAGE", ),
                "control_image": ("IMAGE", ),
                "reference_image": ("IMAGE", ),
                "ipadapter_image": ("IMAGE", ),
                "style_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "batch": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                }),
                "ipadapter_scale":("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "controlnet_conditioning_scale":("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "reference_only":(["disable","enable"],),
                "ip_adapter":(["disable","enable"],),
                "control_net":(["disable","enable"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mainfunc"

    CATEGORY = "LCM_Nodes/nodes"

    def mainfunc(self, prompt: str,steps: int,width:int,height:int,cfg:float,seed: int,image,pipe,batch,strength,reference_image,style_fidelity,ipadapter_scale,reference_only,ip_adapter,ipadapter_image,negative_prompt,control_image,controlnet_conditioning_scale,control_net):
        if ip_adapter =="enable":
            pipe.set_ip_adapter_scale(ipadapter_scale)

        img = image[0].numpy()
        img = img*255.0
        image = Image.fromarray(np.uint8(img))

        img = reference_image[0].numpy()
        img = img*255.0
        reference_image = Image.fromarray(np.uint8(img))
        

        img = control_image[0].numpy()
        img = img*255.0
        control_image = Image.fromarray(np.uint8(img))

        img = ipadapter_image[0].numpy()
        img = img*255.0
        ipadapter_image = Image.fromarray(np.uint8(img))
        
        

        
              
        res = []
        for m in range(batch):
            if control_net == "disable":
                if ip_adapter == "enable" and reference_only=="disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = ipadapter_image,cn_enabled = False,ref_enabled = False).images
                elif reference_only == "disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,cn_enabled = False,ref_enabled = False,ip_enabled = False).images      
                elif reference_only=="enable" and ip_adapter == "disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ref_image=reference_image,style_fidelity=style_fidelity,cn_enabled = False,ip_enabled = False).images      
                else:
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = ipadapter_image,ref_image=reference_image,style_fidelity=style_fidelity,cn_enabled = False).images      
            else:
                if ip_adapter == "enable" and reference_only=="disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = ipadapter_image,controlnet_conditioning_scale=controlnet_conditioning_scale,control_image=control_image,ref_enabled = False).images
                elif reference_only == "disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,controlnet_conditioning_scale=controlnet_conditioning_scale,control_image=control_image,ref_enabled = False,ip_enabled = False).images      
                elif reference_only=="enable" and ip_adapter == "disable":
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ref_image=reference_image,style_fidelity=style_fidelity,control_image=control_image,controlnet_conditioning_scale=controlnet_conditioning_scale,ip_enabled = False).images      
                else:
                    images = pipe(prompt,negative_prompt=negative_prompt,width=width,height=height, image=image, num_inference_steps=steps, strength=strength, guidance_scale=cfg,ip_adapter_image = ipadapter_image,ref_image=reference_image,style_fidelity=style_fidelity,controlnet_conditioning_scale=controlnet_conditioning_scale,control_image=control_image).images      
            
                
            res.append(images[0])                
            
        return (res,)
        
class ImageSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image_1": ("IMAGE",),
                    "image_2": ("IMAGE",),
                    "switch":(["enable","disable"],),
                    }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "load_image"
    def load_image(self, image_1,image_2,switch):
        newarr = [image_1,image_2]
        if switch == "enable":
            newarr = [image_2,image_1]
        return (newarr[0],newarr[1])

class SettingsSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sf_1": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "sf_2": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "sf_3": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "strength_1": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "strength_2": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "strength_3": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "IPAScale_1": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "IPAScale_2": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "IPAScale_3": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "CNScale_1": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "CNScale_2": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "CNScale_3": ("FLOAT",{
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    }),
                    "setting":(["v1","v2","v3"],),
                    }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("FLOAT","FLOAT","FLOAT","FLOAT",)
    FUNCTION = "load_image"
    def load_image(self, sf_1,sf_2,sf_3,strength_1,strength_2,strength_3,IPAScale_1,IPAScale_2,IPAScale_3,CNScale_1,CNScale_2,CNScale_3,setting):
        if setting == "v1":
            newarr = [sf_1,strength_1,IPAScale_1,CNScale_1]
        elif setting == "v2":
            newarr = [sf_2,strength_2,IPAScale_2,CNScale_2]
        elif setting == "v3":
            newarr = [sf_3,strength_3,IPAScale_3,CNScale_3]
        return (newarr[0],newarr[1],newarr[2],newarr[3],)

class FloatNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "Number": ("FLOAT",{
                            "default": 0.6,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.01,
                        }),
                    }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "load_image"
    def load_image(self, Number):
        return (Number,)


class SaveImage_Canvas:
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
            img = image
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            '''data = {
                "lastimage":str(os.path.join(full_output_folder, file)),
                "done":"y"
            }
            json_object = json.dumps(data, indent=4)
            savepath = folder_paths.get_folder_paths("custom_nodes")[0]+"/LCM_Inpaint_Outpaint_Comfy/CanvasToolLone/lastimage.json"
            savepath = Path(savepath)
            with open(savepath, "w") as outfile:
                print(savepath)
                outfile.write(json_object)'''
            p = {
                "data":{
                    "flag":True,
                    "genimg":str(os.path.join(full_output_folder, file))
                }
            }
            data = json.dumps(p).encode('utf-8')
            req = request.Request("http://localhost:5000/getGenStatus",data=data)
            req.add_header("Content-Type","application/json")
            request.urlopen(req)
            
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            
        return { "ui": { "images": results } }


######Copied from WAS_Node_Suite
class ImageResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["rescale", "resize"],),
                "supersample": (["true", "false"],),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resize_width": ("NUMBER",),
                "resize_height": ("NUMBER",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_rescale"

    CATEGORY = "image"

    def image_rescale(self, image, mode="rescale", supersample='true', resampling="lanczos", rescale_factor=2, resize_width=1024, resize_height=1024):
        scaled_images = []
        for img in image:
            scaled_images.append(pil2tensor(self.apply_resize_image(tensor2pil(img), mode, supersample, rescale_factor, resize_width, resize_height, resampling)))
        scaled_images = torch.cat(scaled_images, dim=0)
            
        return (scaled_images, )

    def apply_resize_image(self, image: Image.Image, mode='scale', supersample='true', factor: int = 2, width: int = 1024, height: int = 1024, resample='bicubic'):

        # Get the current width and height of the image
        current_width, current_height = image.size

        # Calculate the new width and height based on the given mode and parameters
        if mode == 'rescale':
            new_width, new_height = int(
                current_width * factor), int(current_height * factor)
        else:
            new_width = width if width % 8 == 0 else width + (8 - width % 8)
            new_height = height if height % 8 == 0 else height + \
                (8 - height % 8)

        # Define a dictionary of resampling filters
        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }
        
        # Apply supersample
        if supersample == 'true':
            image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))

        # Resize the image using the given resampling filter
        resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))
        
        return resized_image



NODE_CLASS_MAPPINGS = {
    "LCMGenerate": LCMGenerate,
    "LoadImageNode_LCM":LoadImageNode_LCM,
    "SaveImage_LCM":SaveImage_LCM,
    "LCM_outpaint_prep":LCM_outpaint_prep,
    "LCMLoader":LCMLoader,
    "LCMLoader_img2img":LCMLoader_img2img,
    "LCMGenerate_img2img": LCMGenerate_img2img,
    "FreeU_LCM":FreeU_LCM,
    "LCMGenerate_ReferenceOnly":LCMGenerate_ReferenceOnly,
    "LCMLoader_ReferenceOnly": LCMLoader_ReferenceOnly,
    "LCMLoader_RefInpaint":LCMLoader_RefInpaint,
    "ImageOutputToComfyNodes":ImageOutputToComfyNodes,
    "ImageShuffle":ImageShuffle,
    "LCMT2IAdapter":LCMT2IAdapter,
    "LCMLoader_controlnet":LCMLoader_controlnet,
    "LCMGenerate_img2img_controlnet":LCMGenerate_img2img_controlnet,
    "LCM_IPAdapter":LCM_IPAdapter,
    "LCMGenerate_img2img_IPAdapter":LCMGenerate_img2img_IPAdapter,
    "LCMGenerate_inpaintv2":LCMGenerate_inpaintv2,
    "LCMLoader_controlnet_inpaint":LCMLoader_controlnet_inpaint,
    "LCM_IPAdapter_inpaint":LCM_IPAdapter_inpaint,
    "LCMGenerate_inpaintv3":LCMGenerate_inpaintv3,
    "OutpaintCanvasTool":OutpaintCanvasTool,
    "stitch":stitch,
    "LCMLora_inpaint":LCMLora_inpaint,
    "LCMLoraLoader_inpaint":LCMLoraLoader_inpaint,
    "LCMLoader_SDTurbo":LCMLoader_SDTurbo,
    "LCMGenerate_SDTurbo":LCMGenerate_SDTurbo,
    "Loader_SegmindVega":Loader_SegmindVega,
    "SegmindVega":SegmindVega,
    "SaveImage_Puzzle":SaveImage_Puzzle,
    "SaveImage_PuzzleV2":SaveImage_PuzzleV2,
    "ImageSwitch":ImageSwitch,
    "SettingsSwitch":SettingsSwitch,
    "FloatNumber":FloatNumber,
    "LCMLora_ipadapter":LCMLora_ipadapter,
    "LCMLoraLoader_ipadapter":LCMLoraLoader_ipadapter,
    "SaveImage_Canvas":SaveImage_Canvas,
    "ComfyNodesToSaveCanvas":ComfyNodesToSaveCanvas,
    "LCMLora_inpaintV2":LCMLora_inpaintV2,
    "ImageDims":ImageDims,
    "ImageResize":ImageResize
}
