import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

DATA_FILE = 'COVID_19.xlsx'
df = pd.read_excel(DATA_FILE)

MAYBE_VARIANTS = [
    'Maybe', 'Maybe ', 'Maybe (можливо)', 'Maybe (можливо) ', 'Maybe (можливо)(можливо)',
    'maybe', 'maybe ', 'maybe (можливо)'
]

def task_inspect_head_info():
    print("=== Перші 5 рядків датасету ===")
    print(df.head())
    print("\n=== Інформація по колонках ===")
    buf = []
    df.info(buf=buf)
    if buf:
        print("\n".join(buf))
    print("\n=== Описова статистика (числові) ===")
    print(df.describe())

def task_parse_dates():
    print("2) Друге завдання — розпарсити колонку 'Date time' та створити індекс 'parsed_date'.\n")
    def parse_date(x):
        return pd.to_datetime(x, dayfirst=True, errors='coerce')

    date_col = "Date time"
    if date_col not in df.columns:
        print("Колонку 'Date time' не знайдено.")
        return

    df.loc[:, 'parsed_date'] = df[date_col].apply(parse_date)
    df.loc[:, 'year'] = df['parsed_date'].dt.year
    df.loc[:, 'month'] = df['parsed_date'].dt.month
    df.loc[:, 'day'] = df['parsed_date'].dt.day
    df.loc[:, 'hour'] = df['parsed_date'].dt.hour
    df.loc[:, 'weekday'] = df['parsed_date'].dt.day_name()
    df.set_index('parsed_date', inplace=True)
    print("Парсинг дат завершено. 'parsed_date' встановлено як індекс. Додані колонки: year, month, day, hour, weekday.")

def task_handle_missing_and_map_bool():
    print("3) Третє завдання — обробка пропущених значень та мапінг Yes/No -> булеві (де можливо).\n")
    before = len(df)
    df.dropna(how='all', inplace=True)
    after = len(df)
    print(f"Видалено повністю порожніх рядків: {before - after}. Залишилось рядків: {after}")

    yes_no_map = {
        'yes': True, 'y': True, 'так': True, 'true': True, '1': True,
        'no': False, 'n': False, 'ні': False, 'false': False, '0': False,
        True: True, False: False, 1: True, 0: False
    }

    cols_to_bool_candidates = [
        'Do you smoke?',
        "Have you had Covid'19 this year?",
        'Have you had influenza this year?',
        'Do you vaccinated influenza?',
        'Do you vaccinated tuberculosis?',
        'Have you had tuberculosis this year?'
    ]

    for col in cols_to_bool_candidates:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        mapped = s.str.lower().map(yes_no_map)
        # Присвоїмо mapped туди, де маємо значення, інакше залишимо оригінал
        df.loc[:, col] = mapped.combine_first(df[col])
        # Тепер гарантовано замінимо 'Maybe' варіанти на NA
        df.loc[:, col] = df[col].replace(MAYBE_VARIANTS + [v.lower() for v in MAYBE_VARIANTS], pd.NA)
        # Якщо після цього в колонці тільки True/False (без NA) — зробимо category
        non_na = df[col].dropna()
        if not non_na.empty and set(non_na.unique()).issubset({True, False}):
            df.loc[:, col] = df[col].astype('category')
        print(f"Колонка '{col}' оброблена. Приклади значень: {df[col].dropna().unique()[:10]}")

def task_convert_types_to_category():
    print("4) Четверте завдання — перетворення деяких колонок у категоріальні типи.\n")
    cat_candidates = ['Gender', 'Region', 'Blood group']
    
    for c in cat_candidates:
        if c not in df.columns:
            print(f"Колонка '{c}' відсутня в датафреймі.")
            continue
        
        # Simple and clean conversion using astype
        df[c] = df[c].astype('category')
        print(f"Колонка '{c}' конвертована в категоріальний тип.")
        
        # Show unique categories for this column
        print(f"Унікальні категорії: {df[c].cat.categories.tolist()}")
        print()
    
    print("Поточні типи колонок:")
    print(df.dtypes)
    print()
    
    # Additional info about categorical columns
    print("Детальна інформація про категоріальні колонки:")
    for col in df.columns:
        if df[col].dtype.name == 'category':
            print(f"{col}: {len(df[col].cat.categories)} категорій - {df[col].cat.categories.tolist()}")



def task_impute_temperature():
    print("5) П'яте завдання — імпутація 'Maximum body temperature'.\n")
    temp_col = 'Maximum body temperature'
    
    if temp_col not in df.columns:
        print("Колонка 'Maximum body temperature' відсутня.")
        return
    
    # Convert to numeric type
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    
    # Count missing values before imputation
    missing_before = int(df[temp_col].isna().sum())
    print(f"Відсутніх значень ДО імпутації: {missing_before}")
    
    # Group-based imputation if Gender column exists
    if 'Gender' in df.columns:
        # Use observed=True to silence the warning and handle only observed categories
        df[temp_col] = df.groupby('Gender', observed=True)[temp_col].transform(
            lambda s: s.fillna(s.median())
        )
        
        missing_after_group = int(df[temp_col].isna().sum())
        print(f"Відсутніх після імпутації медіаною по Gender: {missing_after_group}")
        
        # Show median temperatures by gender for transparency
        gender_medians = df.groupby('Gender', observed=True)[temp_col].median()
        print("Медіанні температури по статі:")
        for gender, median_temp in gender_medians.items():
            print(f"  {gender}: {median_temp:.2f}")
    else:
        print("Колонка 'Gender' відсутня — пропускаємо групову імпутацію.")
        missing_after_group = missing_before
    
    # Overall imputation for remaining missing values
    overall_median = float(df[temp_col].median())
    df[temp_col] = df[temp_col].fillna(overall_median)
    
    missing_after_all = int(df[temp_col].isna().sum())
    print(f"Відсутніх після загальної імпутації: {missing_after_all}. Використана медіана = {overall_median:.2f}")
    
    # Additional statistics
    print(f"\nСтатистика після імпутації:")
    print(f"Мінімальна температура: {df[temp_col].min():.2f}")
    print(f"Максимальна температура: {df[temp_col].max():.2f}")
    print(f"Середня температура: {df[temp_col].mean():.2f}")
    print(f"Стандартне відхилення: {df[temp_col].std():.2f}")

def task_descriptive_stats():
    print("6) Шосте завдання — описова статистика (include='all'):\n")
    print(df.describe(include='all'))

def task_sort_variant1():
    print("7) Сьоме завдання — сортування: Age (зростання), Do you smoke? (спадання)\n")
    sort_cols = []
    if 'Age' in df.columns:
        sort_cols.append('Age')
    else:
        print("Колонка 'Age' не знайдена.")
    if 'Do you smoke?' in df.columns:
        sort_cols.append('Do you smoke?')
    else:
        print("Колонка 'Do you smoke?' не знайдена.")

    if not sort_cols:
        print("Нема колонок для сортування.")
        return
    ascending = [True]
    if len(sort_cols) == 2:
        ascending.append(False)
    df_sorted = df.sort_values(by=sort_cols, ascending=ascending)
    print(f"Перші 10 рядків після сортування по {sort_cols} (ascending={ascending}):")
    print(df_sorted.head(10))

def task_mean_igg_unvaccinated():
    print("8) Восьме завдання — середнє IgG level для невакцинованих від грипу.\n")
    igG_col = 'IgG level'
    vacc_flu_col = 'Do you vaccinated influenza?'
    if igG_col not in df.columns or vacc_flu_col not in df.columns:
        print("Відсутні необхідні колонки для обчислення.")
        return
    condition = (df[vacc_flu_col] == False) | (df[vacc_flu_col] == 'No') | (df[vacc_flu_col] == 0)
    subset = df[condition]
    mean_igg_unvaccinated = pd.to_numeric(subset[igG_col], errors='coerce').mean()
    print(f"Середнє IgG level для невакцинованих від грипу: {mean_igg_unvaccinated}")

def task_frequency_do_you_smoke():
    print("9) Дев'яте завдання — частоти для 'Do you smoke?'\n")
    col = 'Do you smoke?'
    if col not in df.columns:
        print("Колонка відсутня.")
        return
    counts = df[col].value_counts(dropna=False)
    print(f"Частота значень у '{col}':\n{counts}")

def task_visualizations():
    print("10) Десяте завдання — візуалізації (гістограми температури по категоріях).\n")
    temp_col = 'Maximum body temperature'
    plt.figure(figsize=(12, 5))

    # a)
    plt.subplot(1, 2, 1)
    if 'Do you smoke?' in df.columns and temp_col in df.columns:
        for name, group in df.groupby('Do you smoke?'):
            plt.hist(group[temp_col].dropna(), bins=15, alpha=0.6, label=str(name))
        plt.title('Розподіл максимальної температури тіла за "Чи курите?"')
        plt.xlabel('Максимальна температура тіла')
        plt.ylabel('Кількість')
        plt.legend(title='Чи курите?')
    else:
        plt.text(0.5, 0.5, "Відсутні колонки для першого графіка", ha='center')
        plt.axis('off')

    # b)
    plt.subplot(1, 2, 2)
    if 'Have you had influenza this year?' in df.columns and temp_col in df.columns:
        for name, group in df.groupby('Have you had influenza this year?'):
            plt.hist(group[temp_col].dropna(), bins=15, alpha=0.6, label=str(name))
        plt.title('Розподіл максимальної температури тіла за "Чи була грип цього року?"')
        plt.xlabel('Максимальна температура тіла')
        plt.ylabel('Кількість')
        plt.legend(title='Грип цього року?')
    else:
        plt.text(0.5, 0.5, "Відсутні колонки для другого графіка", ha='center')
        plt.axis('off')

    plt.tight_layout()

    backend = matplotlib.get_backend().lower()
    if 'agg' in backend:
        fname = 'temperature_histograms.png'
        plt.savefig(fname)
        print(f"Середовище без інтерактивного вікна — збережено графік у файл: {fname}")
    else:
        plt.show()
        print("Графік показано у вікні.")

def task_full_pipeline():
    print("Запуск повного pipeline (3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 2 -> 10)\n")
    task_handle_missing_and_map_bool()
    task_convert_types_to_category()
    task_impute_temperature()
    task_descriptive_stats()
    task_sort_variant1()
    task_mean_igg_unvaccinated()
    task_frequency_do_you_smoke()
    # parse dates останнім, бо set_index змінює індекс
    if "Date time" in df.columns:
        task_parse_dates()
    task_visualizations()
    print("Pipeline завершено.")


def print_menu():
    print("\nОберіть дію (введіть номер):")
    print("1  - Показати head/info/describe")
    print("2  - Розпарсити дату ('Date time') і зробити індекс ('parsed_date')")
    print("3  - Видалити повністю порожні рядки та замапити Yes/No -> bool")
    print("4  - Перетворити текстові поля у категоріальні (Gender, Region, Blood group)")
    print("5  - Імпутація 'Maximum body temperature' (by Gender + overall)")
    print("6  - Описова статистика (include='all')")
    print("7  - Сортування (Age ASC, Do you smoke? DESC)")
    print("8  - Середнє IgG level для невакцинованих від грипу")
    print("9  - Частоти для 'Do you smoke?'")
    print("10 - Побудувати гістограми (може зберегти файл при lack of display)")
    print("11 - Запустити повний pipeline")
    print("q  - Вихід")

actions = {
    '1': task_inspect_head_info,
    '2': task_parse_dates,
    '3': task_handle_missing_and_map_bool,
    '4': task_convert_types_to_category,
    '5': task_impute_temperature,
    '6': task_descriptive_stats,
    '7': task_sort_variant1,
    '8': task_mean_igg_unvaccinated,
    '9': task_frequency_do_you_smoke,
    '10': task_visualizations,
    '11': task_full_pipeline
}

def main_menu_loop():
    print(f"Дані прочитані з файлу: {DATA_FILE}")
    while True:
        print_menu()
        choice = input("Ваш вибір: ").strip()
        if choice.lower() == 'q':
            print("Вихід.")
            break
        action = actions.get(choice)
        if action is None:
            print("Невірний вибір. Спробуйте ще.")
            continue
        try:
            action()
        except Exception as e:
            print("Під час виконання сталася помилка:", repr(e))

if __name__ == "__main__":
    main_menu_loop()
