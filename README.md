# LCM_Inpaint-Outpaint_Comfy
ComfyUI custom nodes for inpainting/outpainting using the new latent consistency model (LCM)

# Inpaint
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-24%2022-38-53.png'>


# Outpaint
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-24%2022-42-53.png'>


# Prompt Weighting

Note: Requires CPU inference (select CPU in LCMLoader Node). (facing error that i dont know how to fix when using GPU)

Add '+' for more effect and '-' for less effect. Adding more '+' or '-' increases the effects.

<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-26%2020-18-11.png'>



# FreeU
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-26%2020-39-33.png'>


# ReferenceOnly
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-29%2000-09-32.png'>


# Style Transfer
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-29%2020-08-52.png'>


# Image Variations
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-29%2002-33-23.png'>


# Promptless Outpainting
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-29%2003-32-29.png'>


# Image Blending
<img src='https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/blob/main/LCM/Screenshot%20from%202023-10-29%2022-46-43.png'>

# ControlNet/T2I Adapter

T2IAdapter thanks to Michael Poutre https://github.com/M1kep 

Place model folders inside 'ComfyUI/models/controlnet'


# How to Use:
Clone into custom_nodes folder inside your ComfyUI directory
   ```
   git clone https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy
   ```
Install requirements after changing directory to LCM_Inpaint-Outpaint_Comfy folder

```
cd LCM_Inpaint-Outpaint_Comfy
pip install -r requirements.txt
```

Download the model in diffusers format from https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/tree/main and place it inside model/diffusers folder in your ComfyUI directory. (The name of the model folder should be "LCM_Dreamshaper_v7")

Load the workflow by choosing the .json file for inpainting or outpainting.




# Credits

nagolinc's img2img script

patrickvonplaten's StableDiffusionReferencePipeline


