"""Exercise the exported model contract with a small real RGB-D dataset."""
from pathlib import Path
import shutil
import numpy as np
import pytest
import yaml
from object_modeling import build_model, validate_dataset
from object_modeling.core import _write_hull

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'examples/object_003_subset'
CONFIG = ROOT / 'configs/d405_charuco.yaml'


def test_real_rgbd_build_and_model_contract(tmp_path):
    result = build_model(DATA, CONFIG, tmp_path / 'model')
    assert result.point_count > 1000
    assert result.mesh_vertices > 3 and result.mesh_faces > 3
    meta = yaml.safe_load((result.model_dir / 'model.yaml').read_text())
    assert meta['units'] == 'meters'
    assert (result.model_dir / meta['point_cloud']).is_file()
    mesh = result.model_dir / meta['mesh']
    vertices, faces = [], []
    for line in mesh.read_text().splitlines():
        if line.startswith('v '): vertices.append([float(x) for x in line.split()[1:]])
        if line.startswith('f '): faces.append([int(x)-1 for x in line.split()[1:]])
    vertices, faces = np.array(vertices), np.array(faces)
    interior = vertices.mean(axis=0)
    normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]])
    assert np.all(np.einsum('ij,ij->i', normals, vertices[faces].mean(axis=1)-interior) > -1e-12)
    for name in ('model.yaml','quality_report.yaml'):
        assert str(ROOT) not in (result.model_dir / name).read_text()
    report = validate_dataset(DATA)
    assert report.usable_frames == 8


def test_validator_does_not_count_missing_images(tmp_path):
    for name in ('poses.csv', 'camera_info.yaml'):
        shutil.copy2(DATA/name, tmp_path/name)
    report = validate_dataset(tmp_path)
    assert report.usable_frames == 0
    assert any('missing, unreadable' in warning for warning in report.warnings)


def test_degenerate_hull_is_not_reported_as_success(tmp_path):
    with pytest.raises(ValueError, match='hull'):
        _write_hull(tmp_path/'flat.obj', np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]))
    assert not (tmp_path/'flat.obj').exists()


def test_empty_filter_result_is_explicit(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['model']['min_voxel_views'] = 100
    config = tmp_path/'empty.yaml'
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='Too few points'):
        build_model(DATA, config, tmp_path/'model')
    assert not (tmp_path/'model/model.yaml').exists()


def test_invalid_voxel_size(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        build_model(DATA, CONFIG, tmp_path, voxel_size=0)
