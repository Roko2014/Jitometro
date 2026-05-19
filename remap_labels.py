"""
remap_labels.py — Jitometro v2
================================
Remapea las etiquetas YOLO (.txt) del dataset original de 4 clases USDA
a las 3 clases simplificadas SIN modificar las imágenes.

Mapeo:
    Clase original 0 ('3' USDA) → 1 (Semimaduro)
    Clase original 1 ('4' USDA) → 1 (Semimaduro)
    Clase original 2 ('5' USDA) → 2 (Maduro)
    Clase original 3 ('6' USDA) → 2 (Maduro)

Si en el futuro agregas clases USDA 1 y 2 → mapéalas a 0 (Inmaduro).

Uso:
    python remap_labels.py

El script:
  1. Lee los .txt de las carpetas originales (train/valid/test).
  2. Escribe los .txt remapeados en carpetas nuevas (_v2).
  3. NO modifica ni mueve las imágenes originales.
  4. Genera un reporte final con estadísticas de conversión.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

# Raíz de tu dataset original (ajusta si es necesario)
BASE_DIR = Path(r"D:\Jitomate_Rendimiento_IA\Jitomate_python\Jitomate_IA_15Nov2025")

# Carpetas de destino para las etiquetas remapeadas
OUTPUT_DIR = Path(r"D:\Jitomate_Rendimiento_IA\Jitomate_python\Jitomate_IA_v2")

# Splits a procesar
SPLITS = ["train", "valid", "test"]

# ─── TABLA DE REMAPEO ─────────────────────────────────────────────────────────
# Clave: clase original (int)  →  Valor: clase nueva (int)
# Clases originales: 0='3'USDA, 1='4'USDA, 2='5'USDA, 3='6'USDA
# Clases nuevas:     0=Inmaduro, 1=Semimaduro, 2=Maduro
CLASS_REMAP = {
    0: 1,   # '3' USDA → Semimaduro
    1: 1,   # '4' USDA → Semimaduro
    2: 2,   # '5' USDA → Maduro
    3: 2,   # '6' USDA → Maduro
    # Si añades USDA 1 y 2 en el futuro, agrégalos aquí:
    # -1: 0,  # '1' USDA → Inmaduro
    # -2: 0,  # '2' USDA → Inmaduro
}

NAMES_NEW = {0: "Inmaduro", 1: "Semimaduro", 2: "Maduro"}
NAMES_OLD = {0: "3-USDA", 1: "4-USDA", 2: "5-USDA", 3: "6-USDA"}

# ─── LÓGICA PRINCIPAL ─────────────────────────────────────────────────────────

def remap_label_file(src_path: Path, dst_path: Path, remap: dict) -> dict:
    """
    Lee un archivo .txt de etiquetas YOLO y escribe uno nuevo con clases remapeadas.
    Devuelve un dict con conteo de conversiones realizadas.
    """
    counts = defaultdict(int)
    lines_out = []

    with open(src_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        original_class = int(parts[0])

        if original_class not in remap:
            print(f"  ⚠ Clase desconocida {original_class} en {src_path.name} — se omite esta anotación.")
            continue

        new_class = remap[original_class]
        counts[(original_class, new_class)] += 1

        # Reconstruir línea con la nueva clase (bbox sin cambios)
        new_line = f"{new_class} " + " ".join(parts[1:])
        lines_out.append(new_line)

    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    return counts


def process_split(split: str, base_dir: Path, output_dir: Path, remap: dict):
    """Procesa un split completo (train/valid/test)."""
    labels_src = base_dir / split / "labels"
    labels_dst = output_dir / split / "labels"
    images_src = base_dir / split / "images"
    images_dst = output_dir / split / "images"

    if not labels_src.exists():
        print(f"  ⚠ No se encontró la carpeta: {labels_src}")
        return {}

    labels_dst.mkdir(parents=True, exist_ok=True)

    # Crear symlinks o copiar imágenes (symlink para no duplicar espacio en disco)
    if images_src.exists():
        if images_dst.exists():
            shutil.rmtree(images_dst)
        # Intenta symlink; si falla (Windows sin permisos), copia
        try:
            os.symlink(images_src.resolve(), images_dst.resolve())
            print(f"  🔗 Imágenes enlazadas: {images_dst}")
        except (OSError, NotImplementedError):
            shutil.copytree(images_src, images_dst)
            print(f"  📁 Imágenes copiadas: {images_dst}")

    # Procesar etiquetas
    txt_files = list(labels_src.glob("*.txt"))
    total_counts = defaultdict(int)

    for txt_file in txt_files:
        dst_file = labels_dst / txt_file.name
        counts = remap_label_file(txt_file, dst_file, remap)
        for k, v in counts.items():
            total_counts[k] += v

    print(f"  ✅ {len(txt_files)} archivos procesados en '{split}'")
    return total_counts


def print_report(all_counts: dict):
    """Imprime un reporte de conversión consolidado."""
    print("\n" + "="*55)
    print("           REPORTE DE REMAPEO DE CLASES")
    print("="*55)
    print(f"{'Clase original':<20} {'→':<3} {'Clase nueva':<20} {'Anotaciones':>12}")
    print("-"*55)

    grand_total = 0
    merged = defaultdict(int)
    for (old_c, new_c), count in sorted(all_counts.items()):
        label_old = NAMES_OLD.get(old_c, str(old_c))
        label_new = NAMES_NEW.get(new_c, str(new_c))
        print(f"  {label_old:<18} {'→':<3} {label_new:<20} {count:>10,}")
        merged[new_c] += count
        grand_total += count

    print("-"*55)
    print("\nDistribución final por clase nueva:")
    for new_c, count in sorted(merged.items()):
        pct = count / grand_total * 100 if grand_total else 0
        label_new = NAMES_NEW.get(new_c, str(new_c))
        bar = "█" * int(pct / 2)
        print(f"  [{new_c}] {label_new:<12} {count:>6,} anotaciones  ({pct:.1f}%)  {bar}")

    print(f"\n  Total de anotaciones procesadas: {grand_total:,}")
    print("="*55)


def main():
    print(f"\n🍅 Jitometro — Remapeo de Etiquetas v1→v2")
    print(f"   Origen : {BASE_DIR}")
    print(f"   Destino: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_counts = defaultdict(int)

    for split in SPLITS:
        print(f"📂 Procesando split: {split}")
        counts = process_split(split, BASE_DIR, OUTPUT_DIR, CLASS_REMAP)
        for k, v in counts.items():
            all_counts[k] += v

    print_report(all_counts)

    # Copiar data_v2.yaml al directorio de salida como referencia
    yaml_note = (
        "# data_v2.yaml generado automáticamente por remap_labels.py\n"
        f"train: {OUTPUT_DIR / 'train'}\\\n"
        f"val:   {OUTPUT_DIR / 'valid'}\\\n"
        f"test:  {OUTPUT_DIR / 'test'}\\\n\n"
        "nc: 3\n"
        "names: ['Inmaduro', 'Semimaduro', 'Maduro']\n"
    )
    yaml_path = OUTPUT_DIR / "data_v2.yaml"
    yaml_path.write_text(yaml_note)
    print(f"\n📄 data_v2.yaml generado en: {yaml_path}")
    print("\n✅ Remapeo completado. Puedes entrenar con:\n")
    print("   yolo train model=yolo11n.pt data=data_v2.yaml epochs=100 imgsz=640\n")


if __name__ == "__main__":
    main()
