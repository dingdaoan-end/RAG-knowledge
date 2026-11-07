"""只检查前端部署边界；不调用生成 API 或读取 .env。"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_backend_service_is_identical_to_pre_frontend_configuration():
    services = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))['services']
    before = json.loads((ROOT / 'docs/frontend_backend_service_before.json').read_text(encoding='utf-8'))
    assert services['rag-api'] == before


def test_frontend_service_has_no_backend_environment_or_data_volumes():
    service = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))['services']['frontend']
    assert service['build']['context'] == './frontend'
    assert service['ports'] == ['127.0.0.1:8080:8080']
    assert 'env_file' not in service and 'environment' not in service and 'volumes' not in service
    assert service['depends_on']['rag-api']['condition'] == 'service_started'


def test_nginx_uses_same_origin_unbuffered_streams_and_dynamic_docker_dns():
    config = (ROOT / 'frontend/nginx.conf').read_text(encoding='utf-8')
    assert 'location /api/' in config and 'rag-api:8000' in config
    assert 'proxy_buffering off' in config and 'proxy_request_buffering off' in config
    assert 'proxy_read_timeout 300s' in config
    assert 'resolver 127.0.0.11' in config and 'proxy_pass http://$rag_backend' in config
    assert 'try_files $uri $uri/ /index.html' in config
    assert 'Access-Control-Allow-Origin' not in config


def test_frontend_image_contains_static_assets_not_keys_or_node_modules():
    dockerfile = (ROOT / 'frontend/Dockerfile').read_text(encoding='utf-8')
    assert dockerfile.count('FROM ') == 2
    assert 'npm ci' in dockerfile and 'npm run build' in dockerfile
    assert 'COPY --from=build /app/dist' in dockerfile
    assert '.env' not in dockerfile and 'API_KEY' not in dockerfile
    ignored = (ROOT / 'frontend/.dockerignore').read_text(encoding='utf-8')
    assert '.env*' in ignored and 'node_modules/' in ignored
