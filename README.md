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

# How to Use:
Clone into custom_nodes folder inside your ComfyUI directory
   ```
   git clone https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy
   ```
Install requirements after changing directory to LCM_Inpaint-Outpaint_Comfy folder

```
cd LCM_Inpaint-Outpaint_Comfy
pip install -r requirements.txt

Download the model in diffusers format from https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/tree/main and place it inside model/diffusers folder in your ComfyUI directory. (The name of the model folder should be "LCM_Dreamshaper_v7")

Load the workflow by choosing the .json file for inpainting or outpainting.



# Credits
Done by refering to nagolinc's img2img script and the diffuser's inpaint pipeline

