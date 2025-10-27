import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm # Importa o tqdm para a barra de progresso

# --- CONFIGURE AQUI ---

# 1. Defina o caminho para o seu dataset original
# (Baseado na sua imagem, parece ser este)
SOURCE_DIR = Path('../data/all')

# 2. Defina o caminho para onde o novo dataset compactado será salvo
# (Ex: uma nova pasta 'data_compacted')
DEST_DIR = Path('../data/all_compact')

# 3. Defina o tamanho alvo para as imagens (largura, altura)
# Tamanhos comuns para deep learning são (224, 224) ou (256, 256)
TARGET_SIZE = (640, 640)

# 4. Defina a qualidade do JPEG (0-100). 85 é um bom equilíbrio.
JPEG_QUALITY = 85

# --- FIM DA CONFIGURAÇÃO ---

def compact_images(source_dir, dest_dir, target_size, quality):
    """
    Varre o diretório de origem, redimensiona, converte para JPEG
    e salva no diretório de destino, mantendo a estrutura.
    """
    
    # Validação inicial
    if not source_dir.is_dir():
        print(f"Erro: O diretório de origem não existe: {source_dir}")
        return

    # Garante que o diretório de destino exista
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Iniciando compactação...")
    print(f"Origem: {source_dir}")
    print(f"Destino: {dest_dir}")
    print(f"Tamanho Alvo: {target_size}")

    # Encontra todas as imagens (jpg, jpeg, png, bmp, tiff)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    # Usamos rglob para buscar recursivamente em todas as subpastas
    source_files = [f for f in source_dir.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not source_files:
        print("Nenhuma imagem encontrada nos formatos especificados.")
        return

    # Usamos tqdm para criar uma barra de progresso
    for image_path in tqdm(source_files, desc="Processando imagens"):
        try:
            # 1. Calcula o caminho de destino
            # Pega o caminho relativo (ex: BEM_TAB/imagem1.png)
            relative_path = image_path.relative_to(source_dir)
            # Monta o caminho de destino (ex: .../data_compacted/all/BEM_TAB/imagem1.png)
            dest_path = dest_dir / relative_path
            
            # Muda a extensão para .jpg
            dest_path_jpg = dest_path.with_suffix('.jpg')

            # 2. Cria a subpasta de destino se ela não existir
            dest_path_jpg.parent.mkdir(parents=True, exist_ok=True)

            # 3. Abre, redimensiona e salva a imagem
            with Image.open(image_path) as img:
                # Redimensiona a imagem. LANCZOS é um filtro de alta qualidade 
                # para redução (antigamente chamado ANTIALIAS)
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

                # Converte para 'RGB' se for 'RGBA' (PNG com transparência) 
                # ou 'P' (paleta), senão dá erro ao salvar como JPEG
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')

                # Salva a imagem como JPEG com a qualidade definida
                img_resized.save(dest_path_jpg, 
                                 'JPEG', 
                                 quality=quality, 
                                 optimize=True)

        except Exception as e:
            print(f"\nErro ao processar {image_path}: {e}")
            # Continua o processo mesmo se uma imagem falhar

    print(f"\nProcesso concluído! Imagens salvas em {dest_dir}")

# --- Executa a função ---
if __name__ == "__main__":
    compact_images(SOURCE_DIR, DEST_DIR, TARGET_SIZE, JPEG_QUALITY)