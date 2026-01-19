#Я переделал код чтобы он выполнял полностью задание по визуализации рынка недвижимости 
# (раньше не было вычисления средней цены за м² по районам)
# Это единственное чего у меня не было по изначальному заданию.
# Поэтому теперь будет целесообразно сделать индивидуальные правки в одном модуле (без осталных модулей команды)
# Также исправил код в соответствии с вашими замечаниями в фитбеке к проекту 
# Некоторые функции убрал, тк их функционал исполняется в других(новых)(для лучшей интеграции)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЙ КЛАСС ДЛЯ КОНФИГУРАЦИИ
# ════════════════════════════════════════════════════════

class ThemeConfig:
    """Конфигурация оформления для всех графиков"""
    
    @staticmethod
    def get_plotly_layout():
        """Возвращает единую цветовую схему для всех графиков"""
        return {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#333'},
            'plot_bgcolor': 'rgba(240, 240, 240, 0.5)',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
            'hovermode': 'closest'
        }
    
    @staticmethod
    def get_color_scale(n_colors=10):
        """Возвращает цветовую шкалу: красный=дорого, зелёный=дешево"""
        return px.colors.sequential.RdYlGn_r[:n_colors]


class DataValidator:
    """
    Дополнение: Валидация и очистка данных перед анализом
    
    ОБОСНОВАНИЕ UX:
    - Предотвращает ошибки в графиках из-за некорректных данных
    - Позволяет отследить проблемы в исходных данных
    - Улучшает надёжность всей системы
    """
    
    @staticmethod
    def validate_dataset(df):
        """
        Проверяет качество данных и возвращает отчёт
        """
        report = {
            'total_rows': len(df),
            'duplicate_rows': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_issues': {},
            'quality_score': 0
        }
        
        # Проверяем числовые столбцы
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            zero_count = (df[col] == 0).sum()
            report['numeric_issues'][col] = {
                'negative_values': negative_count,
                'zero_values': zero_count
            }
        
        # Вычисляем оценку качества (0-100)
        total_issues = (
            report['duplicate_rows'] +
            sum(report['missing_values'].values()) +
            sum(v.get('negative_values', 0) for v in report['numeric_issues'].values())
        )
        report['quality_score'] = max(0, 100 - (total_issues / len(df) * 100))
        
        return report


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 1: ЦЕНА ЗА М² ПО РАЙОНАМ
# ════════════════════════════════════════════════════════
# Этого раньше не было:
def calculate_price_per_sqm(df):
    """
    Вычисляет среднюю цену за м² по районам
    Возвращает DataFrame и график
    """
    
    df_clean = df[(df['Price'] > 0) & (df['Propertycount'] > 0)].copy()
    df_clean['Price_per_sqm'] = df_clean['Price'] / df_clean['Propertycount']
    
    price_per_sqm = df_clean.groupby('Suburb').agg({
        'Price_per_sqm': ['mean', 'median', 'count', 'std'],
        'Price': 'mean'
    }).round(2)
    
    price_per_sqm.columns = ['Avg_Price_Per_Sqm', 'Median_Price_Per_Sqm', 'Count', 'Std_Dev', 'Avg_Total_Price']
    price_per_sqm = price_per_sqm.sort_values('Avg_Price_Per_Sqm', ascending=False)
    
    price_per_sqm.to_csv('output/07_price_per_sqm_by_suburb.csv')
    
    top_20 = price_per_sqm.head(20)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_20['Avg_Price_Per_Sqm'].values,
        y=top_20.index,
        orientation='h',
        marker_color='lightblue',
        hovertemplate='%{y}<br>Цена за м²: $%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='ТОП-20 районов по средней цене за м²',
        xaxis_title='Цена за м² ($)',
        yaxis_title='Район',
        height=600,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig, price_per_sqm


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 2: РАЗДЕЛЁННАЯ ТЕПЛОВАЯ КАРТА
# ════════════════════════════════════════════════════════

# Исправил(улудшил тепловую карту):
def create_heatmap_split(df):
    """
    Разделяет перегруженную тепловую карту на 4 ясных графика
    """
    
    df_clean = df[(df['Price'] > 0) & (df['Type'].notna())].copy()
    
    type_mapping = {
        'h': 'House (Дом)',
        'u': 'Unit (Квартира)',
        't': 'Townhouse (Таунхаус)'
    }
    
    df_clean['Type_Full'] = df_clean['Type'].map(type_mapping)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Средняя цена по типам жилья',
            'ТОП-10 районов',
            'Количество объявлений',
            'Box plot цен'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'pie'}, {'type': 'box'}]
        ]
    )
    
    # График 1: Средняя цена по типам
    prices_by_type = df_clean.groupby('Type_Full')['Price'].mean().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(
            x=prices_by_type.index,
            y=prices_by_type.values,
            marker_color=['#d62728', '#1f77b4', '#2ca02c'],
            hovertemplate='%{x}<br>Средняя цена: $%{y:,.0f}<extra></extra>',
            name='Цена'
        ),
        row=1, col=1
    )
    
    # График 2: ТОП-10 районов
    prices_by_suburb = df_clean.groupby('Suburb')['Price'].mean().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=prices_by_suburb.values,
            y=prices_by_suburb.index,
            orientation='h',
            marker_color='lightgreen',
            hovertemplate='$%{x:,.0f}<extra></extra>',
            name='Цена района'
        ),
        row=1, col=2
    )
    
    # График 3: Круговая диаграмма
    type_counts = df_clean['Type_Full'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hovertemplate='%{label}<br>%{value} объявлений<extra></extra>',
            name='Количество'
        ),
        row=2, col=1
    )
    
    # График 4: Box plot по типам
    fig.add_trace(
        go.Box(
            y=df_clean['Price'],
            x=df_clean['Type_Full'],
            name='Цена',
            boxmean='sd',
            hovertemplate='%{x}<br>Цена: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text='Тип жилья', row=1, col=1)
    fig.update_yaxes(title_text='Цена ($)', row=1, col=1)
    fig.update_xaxes(title_text='Цена ($)', row=1, col=2)
    fig.update_xaxes(title_text='Тип жилья', row=2, col=2)
    fig.update_yaxes(title_text='Цена ($)', row=2, col=2)
    
    fig.update_layout(
        title_text='РАЗДЕЛЁННАЯ ТЕПЛОВАЯ КАРТА',
        height=900,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 3: ТОП 5 С АДРЕСАМИ
# ════════════════════════════════════════════════════════

# Исправил:
def create_top_bottom_with_addresses(df):
    """
    ТОП 5 самых дорогих и дешевых объявлений
    Показывает АДРЕСА, а не типы жилья
    """
    
    df_clean = df[(df['Price'] > 0) & (df['Address'].notna())].copy()
    
    top_5 = df_clean.nlargest(5, 'Price')
    bottom_5 = df_clean.nsmallest(5, 'Price')
    
    top_5_labels = [f"{addr}({suburb})"
                   for addr, suburb in zip(top_5['Address'], top_5['Suburb'])]
    bottom_5_labels = [f"{addr}({suburb})"
                      for addr, suburb in zip(bottom_5['Address'], bottom_5['Suburb'])]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Топ 5 самых ДОРОГИХ', 'Топ 5 самых ДЕШЕВЫХ'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=top_5['Price'].values,
            y=top_5_labels,
            orientation='h',
            marker_color='darkred',
            hovertemplate='Цена: $%{x:,.0f}<extra></extra>',
            name='Дорогие'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=bottom_5['Price'].values,
            y=bottom_5_labels,
            orientation='h',
            marker_color='darkgreen',
            hovertemplate='Цена: $%{x:,.0f}<extra></extra>',
            name='Дешевые'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Цена ($)', row=1, col=1)
    fig.update_xaxes(title_text='Цена ($)', row=1, col=2)
    
    fig.update_layout(
        title_text='Самые дорогие и дешевые объявления (по АДРЕСАМ)',
        height=600,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 4: ТАБЛИЦА ТИПОВ ЖИЛЬЯ
# ════════════════════════════════════════════════════════

def create_type_mapping_table():
    """Справочник типов жилья"""
    
    mapping_data = {
        'Буква': ['h', 'u', 't'],
        'Русское имя': ['House (Дом)', 'Unit (Квартира)', 'Townhouse (Таунхаус)'],
        'Описание': [
            'Частный дом с территорией',
            'Квартира в многоэтажном доме',
            'Таунхаус (двух-трёхэтажный дом)'
        ]
    }
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Буква', 'Русское имя', 'Описание'],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[mapping_data['Буква'], mapping_data['Русское имя'], mapping_data['Описание']],
            fill_color='lavender',
            align='left',
            font=dict(size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title='Справочник типов жилья',
        height=300,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 5: РАСПРЕДЕЛЕНИЕ ЦЕН
# ════════════════════════════════════════════════════════

def create_price_distribution(df):
    """Гистограмма распределения цен"""
    
    df_clean = df[(df['Price'] > 0)].copy()
    
    Q1 = df_clean['Price'].quantile(0.25)
    Q3 = df_clean['Price'].quantile(0.75)
    IQR = Q3 - Q1
    
    clean_data = df_clean[(df_clean['Price'] >= Q1 - 1.5*IQR) &
                         (df_clean['Price'] <= Q3 + 1.5*IQR)]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clean_data['Price'],
        nbinsx=50,
        marker_color='rgba(33, 128, 141, 0.7)',
        hovertemplate='Цена: $%{x:,.0f}<br>Количество: %{y}<extra></extra>'
    ))
    
    mean_price = clean_data['Price'].mean()
    median_price = clean_data['Price'].median()
    
    fig.add_vline(x=mean_price, line_dash='dash', line_color='red',
                 annotation_text=f'Среднее: ${mean_price:,.0f}')
    fig.add_vline(x=median_price, line_dash='dash', line_color='green',
                 annotation_text=f'Медиана: ${median_price:,.0f}')
    
    fig.update_layout(
        title='Распределение цен на недвижимость',
        xaxis_title='Цена ($)',
        yaxis_title='Количество объявлений',
        height=500,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig


# ════════════════════════════════════════════════════════
# ФУНКЦИЯ 6: МАТРИЦА КОРРЕЛЯЦИЙ
# ════════════════════════════════════════════════════════

def create_correlation_matrix(df):
    """Матрица корреляций между числовыми признаками"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return None, None
    
    numeric_cols = numeric_cols[:8]
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='%{y} vs %{x}<br>Корреляция: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Матрица корреляций',
        height=600,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig, corr_matrix


# ════════════════════════════════════════════════════════
# НОВОЕ ДОПОЛНЕНИЕ 1: АНАЛИЗ РАССЛОЕНИЯ РЫНКА
# ════════════════════════════════════════════════════════

def create_market_segmentation_analysis(df):
    """
    Дополнение: Сегментация рынка по ценовым категориям
    
    ОБОСНОВАНИЕ UX:
    - Пользователи хотят понять структуру рынка
    - Помогает определить целевой сегмент
    - Показывает реальное распределение предложения по ценовым диапазонам
    - Особенно полезно для инвесторов и покупателей
    
    АРХИТЕКТУРА:
    - Интегрируется с существующим ThemeConfig
    - Использует стандартные фильтры данных
    - Возвращает Plotly figure для унифицированного сохранения
    """
    
    df_clean = df[(df['Price'] > 0)].copy()
    
    # Определяем ценовые категории (quartiles)
    price_quartiles = df_clean['Price'].quantile([0.25, 0.5, 0.75])
    
    def categorize_price(price):
        if price <= price_quartiles[0.25]:
            return 'Бюджет (0-25%)'
        elif price <= price_quartiles[0.5]:
            return 'Эконом (25-50%)'
        elif price <= price_quartiles[0.75]:
            return 'Премиум (50-75%)'
        else:
            return 'Люкс (75-100%)'
    
    df_clean['Price_Segment'] = df_clean['Price'].apply(categorize_price)
    
    # Создаём мультиуровневый анализ
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Распределение объявлений по сегментам',
            'Средняя цена по типам в каждом сегменте',
            'Количество объявлений по сегментам (столбцы)',
            'Средняя площадь по сегментам'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ]
    )
    
    # График 1: Пирог распределения сегментов
    segment_counts = df_clean['Price_Segment'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hovertemplate='%{label}<br>%{value} объявлений<extra></extra>'
        ),
        row=1, col=1
    )
    
    # График 2: Средняя цена по типам в каждом сегменте
    segment_type_analysis = df_clean.groupby(['Price_Segment', 'Type'])['Price'].mean().reset_index()
    
    for house_type in segment_type_analysis['Type'].unique():
        data = segment_type_analysis[segment_type_analysis['Type'] == house_type]
        type_names = {'h': 'House', 'u': 'Unit', 't': 'Townhouse'}
        fig.add_trace(
            go.Bar(
                x=data['Price_Segment'],
                y=data['Price'],
                name=type_names.get(house_type, house_type),
                hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # График 3: Столбцы с количеством объявлений
    segment_counts_sorted = segment_counts.reindex([
        'Бюджет (0-25%)',
        'Эконом (25-50%)',
        'Премиум (50-75%)',
        'Люкс (75-100%)'
    ])
    
    fig.add_trace(
        go.Bar(
            x=segment_counts_sorted.index,
            y=segment_counts_sorted.values,
            marker_color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'],
            hovertemplate='%{x}<br>%{y} объявлений<extra></extra>'
        ),
        row=2, col=1
    )
    
    # График 4: Средняя площадь по сегментам
    if 'Propertycount' in df_clean.columns:
        segment_area = df_clean.groupby('Price_Segment')['Propertycount'].mean()
        segment_area = segment_area.reindex([
            'Бюджет (0-25%)',
            'Эконом (25-50%)',
            'Премиум (50-75%)',
            'Люкс (75-100%)'
        ])
        
        fig.add_trace(
            go.Scatter(
                x=segment_area.index,
                y=segment_area.values,
                mode='lines+markers',
                marker=dict(size=10, color='darkblue'),
                hovertemplate='%{x}<br>Средняя площадь: %{y:.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text='Средняя площадь (кв.м)', row=2, col=2)
    
    fig.update_xaxes(title_text='Сегмент', row=1, col=2)
    fig.update_yaxes(title_text='Средняя цена ($)', row=1, col=2)
    fig.update_xaxes(title_text='Сегмент', row=2, col=1)
    fig.update_yaxes(title_text='Количество объявлений', row=2, col=1)
    
    fig.update_layout(
        title_text='АНАЛИЗ РАССЛОЕНИЯ РЫНКА ПО ЦЕНОВЫМ СЕГМЕНТАМ',
        height=900,
        showlegend=True,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig, segment_counts


# ════════════════════════════════════════════════════════
# НОВОЕ ДОПОЛНЕНИЕ 2: АНАЛИЗ ТЕРРИТОРИАЛЬНОГО РАСПРЕДЕЛЕНИЯ
# ════════════════════════════════════════════════════════

def create_geographic_analysis(df):
    """
    Дополнение: Географический анализ рынка
    
    ОБОСНОВАНИЕ UX:
    - Помогает найти лучшие районы для инвестиций
    - Показывает концентрацию рынка
    - Полезно для агентов недвижимости и покупателей
    
    АРХИТЕКТУРА:
    - Расширяет analyse capabilities без изменения ядра
    - Использует существующие фильтры и конфигурацию
    """
    
    df_clean = df[(df['Price'] > 0) & (df['Suburb'].notna())].copy()
    
    # Агрегируем данные по районам
    suburb_analysis = df_clean.groupby('Suburb').agg({
        'Price': ['mean', 'median', 'count'],
        'Propertycount': 'mean'
    }).round(2)
    
    suburb_analysis.columns = ['Avg_Price', 'Median_Price', 'Count', 'Avg_Sqm']
    suburb_analysis = suburb_analysis.sort_values('Count', ascending=False).head(20)
    
    # Создаём комбинированный анализ
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'ТОП-20 районов по количеству объявлений (размер пузыря = цена)',
            'Соотношение средней цены и медианной по районам'
        ),
        specs=[
            [{'type': 'scatter'}],
            [{'type': 'scatter'}]
        ]
    )
    
    # График 1: Пузырьковая диаграмма
    fig.add_trace(
        go.Scatter(
            x=suburb_analysis.index,
            y=suburb_analysis['Count'],
            mode='markers',
            marker=dict(
                size=suburb_analysis['Avg_Price'] / 50000,
                color=suburb_analysis['Avg_Price'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Средняя цена ($)', x=1.15, len=0.4),
                line=dict(width=2, color='white')
            ),
            text=[f"{suburb}<br>Объявлений: {count:.0f}<br>Средняя цена: ${price:,.0f}"
                  for suburb, count, price in zip(
                      suburb_analysis.index,
                      suburb_analysis['Count'],
                      suburb_analysis['Avg_Price']
                  )],
            hovertemplate='%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # График 2: Соотношение цен
    fig.add_trace(
        go.Scatter(
            x=suburb_analysis['Avg_Price'],
            y=suburb_analysis['Median_Price'],
            mode='markers+text',
            marker=dict(size=10, color='#1f77b4'),
            text=suburb_analysis.index,
            textposition='top center',
            hovertemplate='%{text}<br>Средняя: $%{x:,.0f}<br>Медиана: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Добавляем линию y=x для сравнения
    min_price = min(suburb_analysis['Avg_Price'].min(), suburb_analysis['Median_Price'].min())
    max_price = max(suburb_analysis['Avg_Price'].max(), suburb_analysis['Median_Price'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Пересечение (Avg=Median)',
            hovertemplate='Эталонная линия<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text='Район', row=1, col=1)
    fig.update_yaxes(title_text='Количество объявлений', row=1, col=1)
    fig.update_xaxes(title_text='Средняя цена ($)', row=2, col=1)
    fig.update_yaxes(title_text='Медиана цены ($)', row=2, col=1)
    
    # Ротируем метки X на первом графике для читаемости
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    
    fig.update_layout(
        title_text='ГЕОГРАФИЧЕСКИЙ АНАЛИЗ РЫНКА НЕДВИЖИМОСТИ',
        height=900,
        showlegend=False,
        **ThemeConfig.get_plotly_layout()
    )
    
    return fig, suburb_analysis


# ════════════════════════════════════════════════════════
# НОВОЕ ДОПОЛНЕНИЕ 3: СИСТЕМА ГЕНЕРАЦИИ ОТЧЁТОВ
# ════════════════════════════════════════════════════════

def generate_analysis_report(df, output_dir='output'):
    """
    Дополнение: Автоматическая генерация текстового отчёта
    
    ОБОСНОВАНИЕ UX:
    - Пользователь получает выводы в удобном текстовом формате
    - Помогает быстро понять ключевые метрики
    - Дополняет визуализацию текстовым анализом
    
    АРХИТЕКТУРА:
    - Генерирует отчёт в HTML для консистентности
    - Использует те же данные, что и графики
    - Сохраняется отдельно для удобного чтения
    """
    
    df_clean = df[(df['Price'] > 0)].copy()
    
    # Вычисляем ключевые метрики
    metrics = {
        'total_listings': len(df_clean),
        'avg_price': df_clean['Price'].mean(),
        'median_price': df_clean['Price'].median(),
        'std_price': df_clean['Price'].std(),
        'min_price': df_clean['Price'].min(),
        'max_price': df_clean['Price'].max(),
        'price_range': df_clean['Price'].max() - df_clean['Price'].min(),
        'unique_suburbs': df_clean['Suburb'].nunique(),
        'type_distribution': df_clean['Type'].value_counts().to_dict()
    }
    
    # Определяем тренды
    trends = {
        'highest_avg_suburb': df_clean.groupby('Suburb')['Price'].mean().idxmax(),
        'lowest_avg_suburb': df_clean.groupby('Suburb')['Price'].mean().idxmin(),
        'most_listed_type': df_clean['Type'].value_counts().idxmax(),
        'least_listed_type': df_clean['Type'].value_counts().idxmin()
    }
    
    type_names = {'h': 'House', 'u': 'Unit', 't': 'Townhouse'}
    
    # Генерируем HTML отчёт
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Аналитический отчёт по рынку недвижимости</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-box {{
                background-color: #ecf0f1;
                padding: 15px;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
                margin-top: 5px;
            }}
            .trend-box {{
                background-color: #e8f8f5;
                padding: 15px;
                border-left: 4px solid #27ae60;
                border-radius: 4px;
                margin: 10px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            .timestamp {{
                text-align: center;
                color: #95a5a6;
                font-size: 12px;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Аналитический отчёт по рынку недвижимости</h1>
            
            <h2>Ключевые показатели</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value">{metrics['total_listings']:,}</div>
                    <div class="metric-label">Всего объявлений</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['avg_price']:,.0f}</div>
                    <div class="metric-label">Средняя цена</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['median_price']:,.0f}</div>
                    <div class="metric-label">Медиана цены</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['std_price']:,.0f}</div>
                    <div class="metric-label">Стандартное отклонение</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['min_price']:,.0f}</div>
                    <div class="metric-label">Минимальная цена</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['max_price']:,.0f}</div>
                    <div class="metric-label">Максимальная цена</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${metrics['price_range']:,.0f}</div>
                    <div class="metric-label">Диапазон цен</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{metrics['unique_suburbs']}</div>
                    <div class="metric-label">Уникальные районы</div>
                </div>
            </div>
            
            <h2>Тренды и особенности</h2>
            <div class="trend-box">
                <strong>Самый дорогой район:</strong> {trends['highest_avg_suburb']}
            </div>
            <div class="trend-box">
                <strong>Самый дешевый район:</strong> {trends['lowest_avg_suburb']}
            </div>
            <div class="trend-box">
                <strong>Самый популярный тип:</strong> {type_names.get(trends['most_listed_type'], trends['most_listed_type'])}
            </div>
            <div class="trend-box">
                <strong>Самый редкий тип:</strong> {type_names.get(trends['least_listed_type'], trends['least_listed_type'])}
            </div>
            
            <h2>Распределение по типам жилья</h2>
            <table>
                <tr>
                    <th>Тип</th>
                    <th>Кол-во объявлений</th>
                    <th>Процент</th>
                </tr>
    """
    
    total_count = sum(metrics['type_distribution'].values())
    for house_type, count in metrics['type_distribution'].items():
        percentage = (count / total_count) * 100
        type_name = type_names.get(house_type, house_type)
        html_content += f"""
                <tr>
                    <td>{type_name}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <div class="timestamp">
                Отчёт сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем отчёт
    report_path = f'{output_dir}/09_analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path, metrics


# ════════════════════════════════════════════════════════
# НОВОЕ ДОПОЛНЕНИЕ 4: СИСТЕМА ЭКСПОРТА ДАННЫХ
# ════════════════════════════════════════════════════════

def export_detailed_analysis(df, output_dir='output'):
    """
    Дополнение: Расширенный экспорт данных в различные форматы
    
    ОБОСНОВАНИЕ UX:
    - Пользователи хотят работать с данными в Excel/CSV
    - Множественные форматы экспорта повышают гибкость
    - Улучшает интеграцию с другими инструментами
    
    АРХИТЕКТУРА:
    - Модульная система экспорта
    - Использует pandas для работы с данными
    - Легко расширяется для новых форматов
    """
    
    export_info = {}
    
    # 1. Полный датасет
    full_export_path = f'{output_dir}/data_full_export.csv'
    df.to_csv(full_export_path, index=False, encoding='utf-8')
    export_info['full_data'] = full_export_path
    
    # 2. Очищенный датасет (только валидные цены)
    df_clean = df[(df['Price'] > 0) & (df['Price'].notna())].copy()
    clean_export_path = f'{output_dir}/data_cleaned.csv'
    df_clean.to_csv(clean_export_path, index=False, encoding='utf-8')
    export_info['cleaned_data'] = clean_export_path
    
    # 3. Агрегированные данные по районам
    suburb_summary = df_clean.groupby('Suburb').agg({
        'Price': ['count', 'mean', 'median', 'min', 'max', 'std'],
        'Propertycount': 'mean',
        'Type': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    suburb_summary_path = f'{output_dir}/data_suburb_summary.csv'
    suburb_summary.to_csv(suburb_summary_path, encoding='utf-8')
    export_info['suburb_summary'] = suburb_summary_path
    
    # 4. Статистика по типам жилья
    type_summary = df_clean.groupby('Type').agg({
        'Price': ['count', 'mean', 'median', 'std']
    }).round(2)
    
    type_summary_path = f'{output_dir}/data_type_summary.csv'
    type_summary.to_csv(type_summary_path, encoding='utf-8')
    export_info['type_summary'] = type_summary_path
    
    return export_info


# ════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ: СОЗДАНИЕ ВСЕХ ГРАФИКОВ И АНАЛИЗОВ
# ════════════════════════════════════════════════════════

def export_all_visualizations(df, output_dir='output'):
    """
    ГЛАВНАЯ ФУНКЦИЯ: Создаёт все графики и анализы
    Теперь включает новые дополнения для повышения функциональности
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ВИЗУАЛИЗАТОР РЫНКА НЕДВИЖИМОСТИ - РАСШИРЕННАЯ ВЕРСИЯ v3.0")
    print("="*70)
    
    # Валидация данных (НОВОЕ ДОПОЛНЕНИЕ)
    print("\nПроверка качества данных...")
    validator = DataValidator()
    data_report = validator.validate_dataset(df)
    print(f"  Качество данных: {data_report['quality_score']:.1f}%")
    print(f"  Дубликатов: {data_report['duplicate_rows']}")
    print(f"  Проблемных значений: {sum(data_report['numeric_issues'].get(col, {}).get('negative_values', 0) for col in data_report['numeric_issues'])}")
    
    print("\n" + "="*70)
    print("СОЗДАНИЕ БАЗОВЫХ ВИЗУАЛИЗАЦИЙ")
    print("="*70 + "\n")
    
    # 1. Цена за м²
    print("1) Вычисляю цену за м² по районам...")
    fig_price_sqm, price_sqm_data = calculate_price_per_sqm(df)
    fig_price_sqm.write_html(f'{output_dir}/01_price_per_sqm.html')
    print("   [СОХРАНЕНО] 01_price_per_sqm.html")
    
    # 2. Разделённая тепловая карта
    print("2) Создаю разделённую тепловую карту...")
    fig_heatmap_split = create_heatmap_split(df)
    fig_heatmap_split.write_html(f'{output_dir}/02_heatmap_split.html')
    print("   [СОХРАНЕНО] 02_heatmap_split.html")
    
    # 3. ТОП 5 с адресами
    print("3) Создаю график ТОП 5 с адресами...")
    fig_top_bottom = create_top_bottom_with_addresses(df)
    fig_top_bottom.write_html(f'{output_dir}/03_top_bottom_fixed.html')
    print("   [СОХРАНЕНО] 03_top_bottom_fixed.html")
    
    # 4. Распределение цен
    print("4) Создаю график распределения цен...")
    fig_dist = create_price_distribution(df)
    fig_dist.write_html(f'{output_dir}/04_price_distribution.html')
    print("   [СОХРАНЕНО] 04_price_distribution.html")
    
    # 5. Матрица корреляций
    print("5) Создаю матрицу корреляций...")
    fig_corr, corr_matrix = create_correlation_matrix(df)
    if fig_corr:
        fig_corr.write_html(f'{output_dir}/05_correlation.html')
        print("   [СОХРАНЕНО] 05_correlation.html")
    
    # 6. Статистика
    print("6) Создаю таблицу статистики...")
    stats_df = pd.DataFrame({
        'Метрика': [
            'Общее количество объявлений',
            'Средняя цена',
            'Медиана цены',
            'Минимальная цена',
            'Максимальная цена',
            'Стандартное отклонение',
            'Количество районов',
            'Количество типов жилья'
        ],
        'Значение': [
            f"{len(df):,}",
            f"${df['Price'].mean():,.2f}",
            f"${df['Price'].median():,.2f}",
            f"${df['Price'].min():,.2f}",
            f"${df['Price'].max():,.2f}",
            f"${df['Price'].std():,.2f}",
            f"{df['Suburb'].nunique()}",
            f"{df['Type'].nunique()}"
        ]
    })
    
    fig_stats = go.Figure(data=[go.Table(
        header=dict(
            values=['Метрика', 'Значение'],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=12)
        ),
        cells=dict(
            values=[stats_df['Метрика'], stats_df['Значение']],
            fill_color='lavender',
            align='left',
            font=dict(size=11),
            height=25
        )
    )])
    
    fig_stats.update_layout(
        title='Статистическая сводка',
        height=500,
        **ThemeConfig.get_plotly_layout()
    )
    
    fig_stats.write_html(f'{output_dir}/06_statistics.html')
    print("   [СОХРАНЕНО] 06_statistics.html")
    
    # 7. CSV экспорт
    print("7) Экспортирую данные в CSV...")
    print("   [СОХРАНЕНО] 07_price_per_sqm_by_suburb.csv")
    
    # 8. Справочник типов жилья
    print("8) Создаю справочник типов жилья...")
    fig_mapping = create_type_mapping_table()
    fig_mapping.write_html(f'{output_dir}/08_type_mapping.html')
    print("   [СОХРАНЕНО] 08_type_mapping.html")
    
    # НОВЫЕ ДОПОЛНЕНИЯ
    print("\n" + "="*70)
    print("СОЗДАНИЕ РАСШИРЕННЫХ АНАЛИЗОВ (НОВЫЕ ДОПОЛНЕНИЯ)")
    print("="*70 + "\n")
    
    # 9. Анализ расслоения рынка
    print("9) Создаю анализ расслоения рынка...")
    fig_segmentation, segment_data = create_market_segmentation_analysis(df)
    fig_segmentation.write_html(f'{output_dir}/09_market_segmentation.html')
    print("   [СОХРАНЕНО] 09_market_segmentation.html")
    
    # 10. Географический анализ
    print("10) Создаю географический анализ...")
    fig_geo, geo_data = create_geographic_analysis(df)
    fig_geo.write_html(f'{output_dir}/10_geographic_analysis.html')
    print("    [СОХРАНЕНО] 10_geographic_analysis.html")
    
    # 11. Аналитический отчёт
    print("11) Генерирую аналитический отчёт...")
    report_path, metrics = generate_analysis_report(df, output_dir)
    print("    [СОХРАНЕНО] 11_analysis_report.html")
    
    # 12. Детальный экспорт данных
    print("12) Создаю расширенный экспорт данных...")
    export_info = export_detailed_analysis(df, output_dir)
    for name, path in export_info.items():
        print(f"    [СОХРАНЕНО] {path.split('/')[-1]}")
    
    print("\n" + "="*70)
    print("УСПЕШНО ЗАВЕРШЕНО!")
    print("="*70)
    print(f"\nВсего создано файлов: 15+")
    print(f"Основные визуализации: 8")
    print(f"Новые анализы: 3")
    print(f"Экспорт данных: 4+")
    
    print("\n" + "="*70)
    print("СТРУКТУРА ФАЙЛОВ В ПАПКЕ 'output/':")
    print("="*70)
    print("\n[Базовые визуализации]")
    print("  01_price_per_sqm.html           - ТОП районов по цене за м²")
    print("  02_heatmap_split.html           - Разделённая тепловая карта")
    print("  03_top_bottom_fixed.html        - ТОП 5 дорогих и дешёвых")
    print("  04_price_distribution.html      - Распределение цен")
    print("  05_correlation.html             - Матрица корреляций")
    print("  06_statistics.html              - Статистическая сводка")
    print("  08_type_mapping.html            - Справочник типов жилья")
    print("\n[Новые расширенные анализы]")
    print("  09_market_segmentation.html     - Анализ ценовых сегментов")
    print("  10_geographic_analysis.html     - Географический анализ")
    print("  11_analysis_report.html         - Текстовый аналитический отчёт")
    print("\n[Экспорт данных]")
    print("  07_price_per_sqm_by_suburb.csv  - Цены по районам")
    print("  data_full_export.csv            - Полный датасет")
    print("  data_cleaned.csv                - Очищенный датасет")
    print("  data_suburb_summary.csv         - Агрегирование по районам")
    print("  data_type_summary.csv           - Статистика по типам")
    print("\n" + "="*70)
    print("Открой HTML файлы в браузере!\n")


# ════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = pd.read_csv('melb_data.csv')
    export_all_visualizations(df, output_dir='output')