# Release history / История версий

## English

The public repository is [biscet/ModelImprovementLab](https://github.com/biscet/ModelImprovementLab).
Version 1.0.0 starts this source repository's history independently of model
weights and dataset versions.

- `main` contains Model Improvement Lab v3.0.0.
- The `v2.0.1`, `v2.0.0` and `v1.0.0` branches preserve earlier releases.

Before the next release, update VERSION and CHANGELOG and review the intended
source changes. Run the relevant tests and the source privacy check, then commit
the reviewed source and publish it to `main`. When archiving the previous release,
branch from its commit and publish the snapshot alongside `main`.

```bash
git switch main
git branch v3.0.0 main
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
git status --short
git diff --cached
git commit -m "Release Model Improvement Lab v3.1.0"
git push --atomic origin main refs/heads/v3.0.0
```

Stage the intended files before committing. Keep private datasets, weights,
credentials and run artifacts out of Git. Do not force-push over a published
release snapshot.

## Русский

Публичный репозиторий — [biscet/ModelImprovementLab](https://github.com/biscet/ModelImprovementLab).
Версия 1.0.0 начинает историю исходников независимо от версий весов и датасетов.

- В `main` опубликована версия Model Improvement Lab v3.0.0.
- Предыдущие выпуски сохранены в ветках `v2.0.1`, `v2.0.0` и `v1.0.0`.

Перед выпуском обновите VERSION и CHANGELOG, проверьте состав изменений,
необходимые тесты и privacy-check. Сохраните проверенные изменения в коммите
и отправьте его в `main`. Если предыдущий выпуск нужно сохранить, создайте
ветку от его коммита и отправьте обе ветки одним push.
Нужные файлы следует заранее добавить в staging.

Не добавляйте в Git приватные датасеты, веса, ключи и результаты рабочих run.
Не перезаписывайте опубликованные срезы через force-push.
