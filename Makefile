.PHONY: help install lint typecheck format test smoke clean ci

PYTHON := python3

help:
	@echo "MS_Attention4Org 代码质量工具链"
	@echo "================================"
	@echo "  make install    安装开发依赖"
	@echo "  make format     自动格式化代码 (black + isort)"
	@echo "  make lint       运行 linter (flake8 + pylint)"
	@echo "  make typecheck  运行类型检查 (mypy)"
	@echo "  make test       运行测试 (pytest)"
	@echo "  make smoke      运行快速冒烟测试"
	@echo "  make security   运行安全扫描 (bandit)"
	@echo "  make ci         运行完整的 CI 检查链"
	@echo "  make clean      清理缓存和临时文件"

install:
	pip install -r requirements-dev.txt

format:
	$(PYTHON) -m black experiments/ simulation_regression.py
	$(PYTHON) -m isort experiments/ simulation_regression.py

lint:
	$(PYTHON) -m flake8 experiments/ simulation_regression.py --max-line-length=100 --ignore=E501,W503,E741,E731,E402
	$(PYTHON) -m pylint experiments/ simulation_regression.py --score=no

typecheck:
	cd experiments && $(PYTHON) -m mypy exp_framework.py run.py registry.py run_real_experiments.py

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	cd experiments && $(PYTHON) -c "
import sys; sys.path.insert(0, '.')
import exps
from registry import get_experiment
for eid in ['exp01', 'exp02', 'exp03']:
    result = get_experiment(eid).run(n_trials=3, total_budget=50000)
    assert result.get('status') != 'failed', f'{eid} failed'
    print(f'  {eid}: OK')
print('All smoke tests passed')
"

security:
	$(PYTHON) -m bandit -r experiments/ -f json -o bandit-report.json || true
	cat bandit-report.json | $(PYTHON) -m json.tool > /dev/null && echo "Bandit report: bandit-report.json"

ci: format lint typecheck smoke
	@echo "================================"
	@echo "CI 检查链全部通过"
	@echo "================================"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -f bandit-report.json
	rm -rf .pytest_cache
	rm -rf .mypy_cache
