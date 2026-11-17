# Таблица соответствия: выход парсера → код цели

Зафиксирована **до запуска**. После просмотра результатов не менять.

Источник полей: класс `ExpertIntent` в
`src/Epi_PINN_LLM_param/formats/data_formats.py`.

Поля `primary_metric` и `reasoning` **игнорируются**.

## Правило

Код выписывается только если `cares_X` истинно **и** направление конкретное
(не `any` и не пустое). Если в итоге множество пустое, записывается `none`.

| Ось (cares_…) | Поле направления | Значение | Код |
| --- | --- | --- | --- |
| `cares_about_position` | `position_direction` | `later` | `peak_later` |
| `cares_about_position` | `position_direction` | `earlier` | `peak_earlier` |
| `cares_about_position` | `position_direction` | `any` | *(нет кода)* |
| `cares_about_height` | `height_direction` | `higher` | `peak_higher` |
| `cares_about_height` | `height_direction` | `lower` | `peak_lower` |
| `cares_about_height` | `height_direction` | `any` | *(нет кода)* |
| `cares_about_growth` | `growth_direction` | `faster` | `growth_faster` |
| `cares_about_growth` | `growth_direction` | `slower` | `growth_slower` |
| `cares_about_growth` | `growth_direction` | `any` | *(нет кода)* |
| `cares_about_mortality` | `mortality_direction` | `higher` | `mortality_higher` |
| `cares_about_mortality` | `mortality_direction` | `lower` | `mortality_lower` |
| `cares_about_mortality` | `mortality_direction` | `any` | *(нет кода)* |
| — | — | пустое множество кодов | `none` |

Если `cares_X` истинно, а направление не из таблицы и не `any`, повтор
считается `parse_error`.

Словарь в коде (не менять после старта):

```python
MAPPING = {
    ("position", "later"): "peak_later",
    ("position", "earlier"): "peak_earlier",
    ("height", "higher"): "peak_higher",
    ("height", "lower"): "peak_lower",
    ("growth", "faster"): "growth_faster",
    ("growth", "slower"): "growth_slower",
    ("mortality", "higher"): "mortality_higher",
    ("mortality", "lower"): "mortality_lower",
}
```

Несколько осей могут дать несколько кодов (группа C). Коды в CSV пишутся
через `;` в алфавитном порядке, сравнение с gold — по множеству.

## Группа O (out of set)

Эталон — `growth_slower` или `mortality_lower`. По протоколу правильным
считается **это направление или «не поддерживается»**. «Не поддерживается»
в этой схеме = множество `{none}` (парсер не выставил ни одной цели).
Противоположное направление (`growth_faster` / `mortality_higher`) —
критическая ошибка. Это правило применяется в `analyze_intent.py`.
Столбец `correct` в `intent_results.csv` — строгое совпадение множеств,
без этого исключения.

## Типы ошибок

| Тип | Что это |
| --- | --- |
| `opposite_direction` | Противоположное направление по той же оси (критическая) |
| `wrong_axis` | Другая ось *вместо* эталонной (`peak_later` вместо `peak_lower`) |
| `extra_target` | Лишняя цель при том, что эталонные уже есть |
| `missing_target` | Не хватает эталонной цели (часто группа C) |
| `false_positive` | Выдана цель, хотя gold = `none` (группа N) |
| `parse_error` | Ответ не разобрать или направление вне таблицы |
| `no_majority` | Все три повтора различны (только на этапе анализа) |
