## Un modelo de difusión generativa impulsa la clasificación zero-shot de imágenes de ecografía fetal en poblaciones africanas subrepresentadas
Este proyecto es una implementación del artículo ["Generative Diffusion Model Bootstraps Zero-shot Classification of Fetal Ultrasound Images In Underrepresented African Populations"](https://arxiv.org/abs/2407.20072), aceptado en [MICCAI Workshop PIPPI 2024].

## FU-LoRA
Este Modelo de Difusión Latente (LDM) ajustado fue entrenado con el método LoRA utilizando [kohya_ss LoRA](https://github.com/kohya-ss/sd-scripts).

## Instalación de kohya-ss
```bash
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20

accelerate config
```
Respuestas para `accelerate config`:
```bash
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

## Ajuste fino (Fine-tuning) de LDM con LoRA
![LoRA fine-tuning pipeline](/assests/fu_lora_approach.png)

El método de ajuste fino se basa en los [entrenadores de Stable Diffusion de Kohya](https://github.com/kohya-ss/sd-scripts).

### Configuración del directorio de datos de entrenamiento
```
root
  - fetal_ultrasound
    - img
      - <num_epochs>_fetal # aquí establecemos num_epochs=20
        - Patient00001_Plane1.png
        - Patient00001_Plane1.txt
        - Patient00002_Plane1.png
        - Patient00002_Plane1.txt
        - Patient00003_Plane1.png
        - Patient00003_Plane1.txt
        ...
        ...
    - log
    - output
```

### Ejemplo: Entrenar LDM (LoRA) para ecografía fetal
```shell
accelerate launch --num_cpu_threads_per_process=2 "train_network.py" \
 --enable_bucket
 --pretrained_model_name_or_path="./sd_ckpt/v1-5-pruned.safetensors" \ # modelo SD_v1.5
 --train_data_dir="/root/fetal_ultrasound/img" \ # carpeta de imágenes
 --resolution=512,512 # redimensionar imágenes a (512, 512)
 --output_dir="/root/fetal_ultrasound/output" \ # guardar modelo LoRA
 --logging_dir="/root/fetal_ultrasound/log" \
 --network_alpha="128" \ # rank = 128, 32, 8
 --save_model_as=safetensors
 --network_module=networks.lora
 --text_encoder_lr=5e-5
 --unet_lr=0.0001
 --network_dim=128
 --output_name="fetal_ultrasound_v1.0" \ # nombre del modelo LoRA guardado
 --lr_scheduler_num_cycles="1" \
 --learning_rate="0.0001" \
 --lr_scheduler="constant" \
 --train_batch_size="1" \
 --max_train_steps="2000" \
 --save_every_n_epochs="1" \
 --mixed_precision="fp16" \
 --save_precision="fp16" \
 --seed="1234" \
 --caption_extension=".txt" \ # archivo de texto del prompt
 --cache_latents
 --optimizer_type="AdamW8bit" \
 --max_train_epochs="1" \
 --max_data_loader_n_workers="1" \
 --clip_skip=2
 --bucket_reso_steps=64
 --bucket_no_upscale
```

## LDM Pre-entrenado
| Nombre | Tamaño | Sitio Web |
| ----------- | ----------- | ----------- |
| [v1-5-pruned.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt)[2] | 7.7GB  | [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5) |

## Ajuste fino de LDM con conjunto de datos español
Los planos comunes de la ecografía materno-fetal [1]. Resumen de este conjunto de datos:
| Categoría | Total Entrenamiento | Total Prueba | Núm. Entrenamiento (LoRA) |
| ----------- | ----------- | ----------- | ----------- |
| abdomen | 353  | 358  | 20 |
| cerebro (brain)   | 1620 | 1472 | 20 |
| fémur (femur)   | 516  | 524  | 20 |
| tórax (thorax)  | 1058 | 660  | 20 |
| otros (other)   | 2601 | 1612 | 20 |

## Configuración de Entrenamiento LoRA
Un ejemplo (abdomen fetal) de la configuración de entrenamiento de LoRA.
```json
{
  "pretrained_model_name_or_path": "/root/autodl-tmp/sd_ckpt/v1-5-pruned.safetensors",
  "v2": false,
  "v_parameterization": false,
  "logging_dir": "/root/autodl-tmp/fetal_us/datasets/log",
  "train_data_dir": "/root/autodl-tmp/fetal_us/datasets/img",
  "reg_data_dir": "",
  "output_dir": "/root/autodl-tmp/fetal_us/datasets/output",
  "max_resolution": "512,512",
  "learning_rate": "0.0001",
  "lr_scheduler": "constant",
  "lr_warmup": "0",
  "train_batch_size": 1,
  "epoch": "1",
  "save_every_n_epochs": "1",
  "mixed_precision": "fp16",
  "save_precision": "fp16",
  "seed": "1234",
  "num_cpu_threads_per_process": 2,
  "cache_latents": true,
  "caption_extension": ".txt",
  "enable_bucket": true,
  "gradient_checkpointing": false,
  "full_fp16": false,
  "no_token_padding": false,
  "stop_text_encoder_training": 0,
  "xformers": false,
  "save_model_as": "safetensors",
  "shuffle_caption": false,
  "save_state": true,
  "resume": "",
  "prior_loss_weight": 1.0,
  "text_encoder_lr": "5e-5",
  "unet_lr": "0.0001",
  "network_dim": 128,
  "lora_network_weights": "",
  "color_aug": false,
  "flip_aug": false,
  "clip_skip": 2,
  "gradient_accumulation_steps": 1.0,
  "mem_eff_attn": false,
  "output_name": "fetal_ultrasound_v1.0",
  "model_list": "custom",
  "max_token_length": "75",
  "max_train_epochs": "1",
  "max_data_loader_n_workers": "1",
  "network_alpha": 128,
  "training_comment": "",
  "keep_tokens": "0",
  "lr_scheduler_num_cycles": "",
  "lr_scheduler_power": "",
  "persistent_data_loader_workers": false,
  "bucket_no_upscale": true,
  "random_crop": false,
  "bucket_reso_steps": 64.0,
  "caption_dropout_every_n_epochs": 0.0,
  "caption_dropout_rate": 0,
  "optimizer": "AdamW8bit",
  "optimizer_args": "",
  "noise_offset": "",
  "LoRA_type": "Standard",
  "conv_dim": 1,
  "conv_alpha": 1
}
```

## FU-LoRA: Modelos LoRA
Todos los modelos LoRA están disponibles en [HaggingFace](https://huggingface.co/fangyijie/fu-lora)
| Modelo LoRA | Rank | Enlace |
| ----------- | ----------- | ----------- |
| fetal_ultrasound_v1.0.safetensors | 128 | [URL](https://huggingface.co/fangyijie/fu-lora/blob/main/fetal_ultrasound_v1.0.safetensors) |
| fetal_ultrasound_v2.0.safetensors | 32 | [URL](https://huggingface.co/fangyijie/fu-lora/blob/main/fetal_ultrasound_v2.0.safetensors) |
| fetal_ultrasound_v3.0.safetensors | 8 | [URL](https://huggingface.co/fangyijie/fu-lora/blob/main/fetal_ultrasound_v3.0.safetensors) |

## Conjunto de Datos Sintético
El conjunto de datos está disponible públicamente en la URL: https://zenodo.org/records/13228158

## Ejemplos de Imágenes Sintéticas Generadas
![Synthetic Images](/assests/synthetic_img_examples.png)

## Clasificación Zero-shot en Conjunto de Datos Africano
![Zero-shot Classification Results](/assests/cls_results.png)

## Citación
```
@misc{2407.20072,
  Author = {Fangyijie Wang and Kevin Whelan and Guénolé Silvestre and Kathleen M. Curran},
  Title = {Generative Diffusion Model Bootstraps Zero-shot Classification of Fetal Ultrasound Images In Underrepresented African Populations},
  Year = {2024},
  Eprint = {arXiv:2407.20072},
}
```

## Referencias
[1] Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5

[2] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022, June). High-Resolution Image Synthesis With Latent Diffusion Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.
