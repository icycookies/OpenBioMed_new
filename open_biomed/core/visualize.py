from typing import List, Optional, Union

from datetime import datetime
from pymol import cmd
import os
import shutil
import imageio
from rdkit.Chem import Draw, rdDepictor

from open_biomed.data import Molecule, Protein
from open_biomed.utils.config import Config, merge_config

def convert_png2gif(png_dir, gif_file, duration=0.5):
    """
    generate gif from png
    :param png_file
    :param gif_file
    """
    frames = [imageio.imread(os.path.join(png_dir, f)) for f in sorted(os.listdir(png_dir)) if f.endswith(".png")]
    imageio.mimsave(gif_file, frames, duration=duration)

def visualize_complex_3D(
    file: str, 
    protein_file: Optional[str]=None, 
    ligand_file: Optional[str]=None,
    config: Config=None,
    rotate=False,
    num_frames=20
):
    cmd.reinitialize()
    cmd.bg_color(getattr(config, "background_color", "white"))
    cmd.set("ray_opaque_background", getattr(config, "ray_opaque_background", 1))
    
    if ligand_file is not None:
        cmd.load(ligand_file, "ligand")
        for elem in config.molecule.show:
            cmd.show(elem, "ligand")
        for elem in config.molecule.__dict__.keys():
            if elem not in ["show", "mode"]:
                cmd.set(elem, config.molecule.__dict__[elem], "ligand")
        cmd.orient("ligand")
    if protein_file is not None:
        cmd.load(protein_file, "protein")
        cmd.hide("everything", "protein")
        for elem in config.protein.show:
            cmd.show(elem, "protein")
        for elem in config.protein.__dict__.keys():
            if elem not in ["color", "show", "cnc"]:
                cmd.set(elem, config.protein.__dict__[elem], "protein")
        cmd.color(getattr(config.protein, "color", "grey"), "protein")
        if getattr(config.protein, "cnc", False):
            cmd.util.cnc("protein")
        cmd.orient("protein")

    cmd.zoom("all")
    
    if rotate:
        cmd.mset(f"1 x{num_frames}")
        cmd.util.mroll(1, num_frames, 360 // num_frames)

        name_dir = os.path.dirname(file)
        name_base = os.path.basename(file)
        name_time = datetime.now()
        name_temp = os.path.join(name_dir, "rotate_png_"+name_time.strftime("%Y%m%d_%H%M%S")+"_"+name_base[:-4])
        if not os.path.exists(name_temp):
            os.makedirs(name_temp)
        cmd.mpng(f"{name_temp}/", width=config.width, height=config.height)

        convert_png2gif(name_temp, file)
        if os.path.exists(name_temp):
            shutil.rmtree(name_temp)
    else:
        cmd.png(file, width=config.width, height=config.height, dpi=config.dpi)

def visualize_protein_with_pocket(
    file: str,
    protein_file: str=None, 
    pocket_indices: List[int]=None,
    config: Config=None,
    rotate=False,
    num_frames=50
) -> str:
    cmd.reinitialize()
    cmd.bg_color(getattr(config, "background_color", "white"))
    cmd.set("ray_opaque_background", getattr(config, "ray_opaque_background", 1))
    
    cmd.load(protein_file, "protein")
    cmd.hide("everything", "protein")
    cmd.show("surface", "protein")
    residues = "+".join([str(elem) for elem in pocket_indices])
    cmd.select("highlight", f"protein and resi {residues}")
    
    # Color the selected residues with the chosen highlight color
    cmd.color("red", "highlight")
    cmd.color("grey", "protein and not highlight")
    
    cmd.set("transparency", 0.3, "protein and not highlight")

    if rotate:
        cmd.mset(f"1 x{num_frames}")
        cmd.util.mroll(1, num_frames, 360 // num_frames)

        name_dir = os.path.dirname(file)
        name_base = os.path.basename(file)
        name_time = datetime.now()
        name_temp = os.path.join(name_dir, "rotate_png_"+name_time.strftime("%Y%m%d_%H%M%S")+"_"+name_base[:-4])
        if not os.path.exists(name_temp):
            os.makedirs(name_temp)
        cmd.mpng(f"{name_temp}/", width=config.width, height=config.height)

        convert_png2gif(name_temp, file)
        if os.path.exists(name_temp):
            shutil.rmtree(name_temp)
    else:
        cmd.png(file, width=config.width, height=config.height, dpi=config.dpi)

    return file

class MoleculeVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, molecule: Molecule, config: Union[str, Config]="2D", rotate: bool=False) -> str:
        # img_file_type = "gif" if rotate else "png"
        img_file_type = "png"
        molecule._add_name()
        img_file = f"./tmp/{molecule.name}.{img_file_type}"
        if isinstance(config, str):
            cfg_path = f"./configs/visualization/molecule/{config}.yaml"
            config = merge_config(
                Config(cfg_path),
                Config("./configs/visualization/global_config.yaml")
            )
        if config.molecule.mode == "3D":
            sdf_file = molecule.save_sdf()
            visualize_complex_3D(img_file, ligand_file=sdf_file, config=config, rotate=rotate)
        if config.molecule.mode == "2D":
            molecule._add_rdmol()
            rdDepictor.Compute2DCoords(molecule.rdmol)
            Draw.MolToImageFile(molecule.rdmol, img_file, size=(config.width, config.height))

        return os.path.abspath(img_file)

class ProteinVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, protein: Protein, config: Union[str, Config]="cartoon", rotate: bool=False) -> str:
        pdb_file = protein.save_pdb()
        # img_file_type = "gif" if rotate else "png"
        img_file_type = "png"
        img_file = f"./tmp/{protein.name}.{img_file_type}"

        if isinstance(config, str):
            cfg_path = f"./configs/visualization/protein/{config}.yaml"
            config = merge_config(
                Config(cfg_path),
                Config("./configs/visualization/global_config.yaml")
            )

        visualize_complex_3D(img_file, protein_file=pdb_file, config=config, rotate=rotate)

        return os.path.abspath(img_file)

class ComplexVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, 
        molecule: Molecule, 
        protein: Protein, 
        molecule_config: Union[str, Config]="ball_and_stick", 
        protein_config: Union[str, Config]="cartoon", 
        rotate: bool=True
    ) -> str:
        # img_file_type = "gif" if rotate else "png"
        img_file_type = "png"
        sdf_file = molecule.save_sdf()
        pdb_file = protein.save_pdb()
        img_file = f"./tmp/complex_{molecule.name}_{protein.name}.{img_file_type}"
        if isinstance(molecule_config, str):
            cfg_path = f"./configs/visualization/molecule/{molecule_config}.yaml"
            molecule_config = Config(cfg_path)
        if isinstance(protein_config, str):
            cfg_path = f"./configs/visualization/protein/{protein_config}.yaml"
            protein_config = Config(cfg_path)
        config = merge_config(merge_config(molecule_config, protein_config), Config("./configs/visualization/global_config.yaml"))

        visualize_complex_3D(img_file, ligand_file=sdf_file, protein_file=pdb_file, config=config, rotate=rotate)

        return os.path.abspath(img_file)