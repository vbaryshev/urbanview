# -*- coding: utf-8 -*-
# UrbanViewWidget (v3.3.0): RL стабильность, метрики и индикатор обучения
# - Фикс нормализации q_values (jsonb + защита)
# - Сокращённое состояние (без name_project/tags) для переиспользования
# - Метрики RL + дата и индикатор последнего обучения
# - Q-таблица: updated_at, rl_meta: last_train_ts/last_train_updates
# Требует: QGIS 3.x, psycopg2, matplotlib, PostgreSQL JSONB

import os
import sys
import csv
import json
import math
import time
import uuid
import shutil
import random
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras as pgextras
except Exception as e:
    raise RuntimeError("psycopg2 не установлен в окружении QGIS. Установите psycopg2-binary. Ошибка: %s" % e)

# Matplotlib
HAS_MPL = True
MPL_IMPORT_ERROR = ''
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
except Exception as e:
    HAS_MPL = False
    MPL_IMPORT_ERROR = str(e)

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt import QtXml
from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsVectorLayer,
    QgsFeatureRequest,
    QgsReadWriteContext,
    QgsPrintLayout,
    QgsLayoutItemPicture,
    QgsLayoutItemLabel,
    QgsLayoutExporter,
    QgsRectangle
)
from qgis.gui import QgsMapToolEmitPoint, QgsMapCanvas, QgsVertexMarker

# --- Конфигурация подключения и путей ---
PG = dict(host='localhost', port=5432, dbname='', user='', password='')
SCHEMA = 'urbanview'

QPT_PATH = r'/home/geonode/Рабочий стол/Неделя_1/mephi/RL/urban_view/data/urban_view.qpt'
PDF_OUT_DIR = r'/home/geonode/Рабочий стол/Неделя_1/mephi/RL/urban_view/data/output/urban_viewout_pdf'
DATASET_IMG_DIR = r'/home/geonode/Рабочий стол/Неделя_1/mephi/RL/urban_view/data/dataset_image'
DATASET_META_CSV = r'/home/geonode/Рабочий стол/Неделя_1/mephi/RL/urban_view/data/dataset_imag.csv'

# --- Утилиты ---
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

ensure_dir(PDF_OUT_DIR)
ensure_dir(DATASET_IMG_DIR)
ensure_dir(Path(DATASET_META_CSV).parent)

def open_in_file_manager(path):
    if not path:
        return
    path = str(path)
    if os.name == 'nt':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.call(['open', path])
    else:
        subprocess.call(['xdg-open', path])

# --- PostgreSQL помощник ---
class Pg:
    def __init__(self, conn_params):
        self.conn_params = conn_params
        self.conn = psycopg2.connect(**conn_params)
        self.conn.autocommit = False

    def cursor(self):
        return self.conn.cursor(cursor_factory=pgextras.RealDictCursor)

    def fetchall(self, q, args=None):
        with self.cursor() as cur:
            cur.execute(q, args or [])
            return cur.fetchall()

    def fetchone(self, q, args=None):
        with self.cursor() as cur:
            cur.execute(q, args or [])
            return cur.fetchone()

    def execute(self, q, args=None, commit=True, returning=False):
        with self.cursor() as cur:
            cur.execute(q, args or [])
            res = cur.fetchone() if returning else None
            if commit:
                self.conn.commit()
            return res

    def execute_rowcount(self, q, args=None, commit=True):
        with self.cursor() as cur:
            cur.execute(q, args or [])
            rc = cur.rowcount
            if commit:
                self.conn.commit()
            return rc

    def many(self, statements):
        with self.cursor() as cur:
            for q, args in statements:
                cur.execute(q, args or [])
            self.conn.commit()

    def commit(self):
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except:
            pass

# --- RL агент ---
class RLAgent:
    def __init__(self, db: Pg, schema: str, alpha=0.4, gamma=0.0, epsilon=0.0):
        # alpha чуть выше, gamma=0 (многорукий бандит), epsilon=0 (детерминированные рекомендации)
        self.db = db
        self.schema = schema
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._q_keys = ('1', '2', '3')

    # ---- Метаданные (rl_meta) ----
    def _meta_get(self, key):
        row = self.db.fetchone(f"SELECT value FROM {self.schema}.rl_meta WHERE key=%s", [key])
        return row['value'] if row and 'value' in row else None

    def _meta_set(self, key, value):
        self.db.execute(f"""
            INSERT INTO {self.schema}.rl_meta(key, value) VALUES (%s,%s)
            ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
        """, [key, str(value)], commit=True)

    def last_train_at(self):
        v = self._meta_get('last_train_ts')
        return v

    def last_train_updates(self):
        v = self._meta_get('last_train_updates')
        try:
            return int(v)
        except:
            return None

    # ---- Нормализация и хэш состояния ----
    @staticmethod
    def _hash_state(context_dict: dict) -> str:
        # Хэшируем только ключи с детерминированной сортировкой
        s = json.dumps(context_dict, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _normalize_qv(self, val) -> dict:
        # Приводим q_values к словарю с ключами '1','2','3' и значениями float
        qv = {}
        try:
            if isinstance(val, dict):
                qv = val
            elif isinstance(val, str):
                qv = json.loads(val) if val.strip() else {}
            elif isinstance(val, (list, tuple)):
                if len(val) >= 3:
                    qv = {'1': val[0], '2': val[1], '3': val[2]}
                else:
                    qv = {}
            else:
                qv = {}
        except Exception:
            qv = {}
        fixed = {}
        for k in ('1', '2', '3'):
            v = None
            if isinstance(qv, dict):
                v = qv.get(k, qv.get(int(k), None)) if qv else None
            try:
                fixed[k] = float(v) if v is not None else 0.0
            except Exception:
                fixed[k] = 0.0
        return fixed

    def _get_or_init_q(self, state_hash: str):
        row = self.db.fetchone(f"SELECT id, q_values, visit_count FROM {self.schema}.rl_q_table WHERE state_hash=%s", [state_hash])
        if row:
            qv = self._normalize_qv(row.get('q_values'))
            vc = row.get('visit_count') or 0
            return row['id'], qv, vc
        qv_new = {'1': 0.0, '2': 0.0, '3': 0.0}
        self.db.execute(
            f"INSERT INTO {self.schema}.rl_q_table(state_hash, q_values, visit_count, updated_at) VALUES (%s,%s,%s, NOW())",
            [state_hash, json.dumps(qv_new, ensure_ascii=False), 0], commit=True
        )
        row = self.db.fetchone(f"SELECT id, q_values, visit_count FROM {self.schema}.rl_q_table WHERE state_hash=%s", [state_hash])
        if row:
            return row['id'], self._normalize_qv(row.get('q_values')), row.get('visit_count') or 0
        return None, qv_new, 0

    def get_q_values(self, context: dict):
        h = self._hash_state(context)
        _, qv, vc = self._get_or_init_q(h)
        return qv, vc, h

    def reset_state(self, context: dict):
        h = self._hash_state(context)
        self.db.execute(f"DELETE FROM {self.schema}.rl_q_table WHERE state_hash=%s", [h], commit=True)

    def update(self, context: dict, action: int, reward: float):
        state_hash = self._hash_state(context)
        rid, qv, vc = self._get_or_init_q(state_hash)
        akey = str(action)
        old_q = float(qv.get(akey, 0.0))
        # Бандит: целевое значение = непосредственная награда
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward)
        qv[akey] = new_q
        self.db.execute(
            f"UPDATE {self.schema}.rl_q_table SET q_values=%s, visit_count=COALESCE(visit_count,0)+1, updated_at=NOW() WHERE state_hash=%s",
            [json.dumps(qv, ensure_ascii=False), state_hash], commit=False
        )
        return new_q, abs(new_q - old_q)

    def suggest_priorities_for_images(self, project_context: dict, image_features: list):
        # Ранжируем по Q(a=1): кому «выгоднее» дать приоритет 1, тот выше
        scored = []
        for idx, feats in enumerate(image_features):
            feats = feats if isinstance(feats, dict) else {}
            ctx = dict(project_context, **feats)
            qv, _, _ = self.get_q_values(ctx)
            score = float(qv.get('1', 0.0)) if isinstance(qv, dict) else 0.0
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        priorities = [None] * len(image_features)
        for rank, (idx, _) in enumerate(scored):
            priorities[idx] = min(rank + 1, 3)
        return priorities

    def train_from_history(self):
        # Загружаем историю (из project_image + project_card)
        rows = self.db.fetchall(f"""
            SELECT pi.id as image_id, pi.priority, pi.path, pi.project_id,
                   pc.status, pc.mean_level, pc.mean_height, pc.density,
                   pc.area, pc.footprint, pc.spp, pc.year_entry,
                   fno.name as fno_group, vo.name as view_object, mt.name as morphotype,
                   ty.name as typology, vr.name as vri
            FROM {self.schema}.project_image pi
            JOIN {self.schema}.project_card pc ON pc.id = pi.project_id
            LEFT JOIN {self.schema}.fno_group fno ON fno.id = pc.id_fno_group
            LEFT JOIN {self.schema}.view_object vo ON vo.id = pc.id_view_object
            LEFT JOIN {self.schema}.morphotype mt ON mt.id = pc.id_morphotype
            LEFT JOIN {self.schema}.typology ty ON ty.id = pc.id_typology
            LEFT JOIN {self.schema}.vri vr ON vr.id = pc.vri_id
        """)
        # Простые фичи изображения (коэффициент сторон — бины)
        def image_feats(path):
            try:
                qimg = QtGui.QImage(path)
                w, h = qimg.width(), qimg.height()
                if w <= 0 or h <= 0:
                    return {'ar_bucket': 'unk'}
                ar = w / float(h)
                if ar < 0.8:
                    b = 'tall'
                elif ar > 1.4:
                    b = 'wide'
                else:
                    b = 'squareish'
                return {'ar_bucket': b}
            except Exception:
                return {'ar_bucket': 'unk'}

        updates = 0
        unique_states = set()
        sum_delta = 0.0

        # В одной транзакции, чтобы быстрее
        try:
            for r in rows:
                # Контекст: только устойчивые категориальные + бины числовых
                ctx = {
                    'fno_group': r['fno_group'] or '',
                    'view_object': r['view_object'] or '',
                    'morphotype': r['morphotype'] or '',
                    'typology': r['typology'] or '',
                    'vri': r['vri'] or '',
                    'status': r['status'] or '',
                    'area_b': self._bucket(r['area']),
                    'spp_b': self._bucket(r['spp']),
                    'mean_level_b': self._bucket(r['mean_level']),
                    'mean_height_b': self._bucket(r['mean_height']),
                    'density_b': self._bucket(r['density']),
                    'year_b': self._year_bucket(r['year_entry'])
                }
                feats = image_feats(r['path'])
                full_ctx = dict(ctx, **feats)
                # Награда по сохранённому приоритету
                pr = int(r['priority'] or 3)
                reward = 1.0 if pr == 1 else (0.5 if pr == 2 else 0.33)
                _, d = self.update(full_ctx, pr, reward)
                updates += 1
                sum_delta += d
                unique_states.add(self._hash_state(full_ctx))
            # Коммит одной пачкой
            self.db.commit()
        except Exception:
            # На всякий случай
            self.db.commit()

        # Сохраним метаданные обучения
        self._meta_set('last_train_ts', datetime.now().isoformat(timespec='seconds'))
        self._meta_set('last_train_updates', updates)
        return {
            'updates': updates,
            'unique_states': len(unique_states),
            'avg_delta': (sum_delta / updates) if updates > 0 else 0.0
        }

    def get_metrics(self, current_contexts=None):
        # Общие метрики по Q-таблице
        agg = self.db.fetchone(f"""
            SELECT
                COUNT(*)::int AS n_states,
                COALESCE(AVG(visit_count),0)::float AS avg_visits,
                COALESCE(SUM(visit_count),0)::bigint AS sum_visits,
                COALESCE(AVG(ABS(COALESCE((q_values->>'1')::numeric,0))),0)::float AS mean_abs_q1,
                COALESCE(AVG(ABS(COALESCE((q_values->>'2')::numeric,0))),0)::float AS mean_abs_q2,
                COALESCE(AVG(ABS(COALESCE((q_values->>'3')::numeric,0))),0)::float AS mean_abs_q3,
                COALESCE(MAX(updated_at), NULL) AS last_update
            FROM {self.schema}.rl_q_table
        """) or {}
        # Покрытие текущего набора изображений (если передан)
        coverage = None
        if current_contexts:
            known = 0
            for ctx in current_contexts:
                h = self._hash_state(ctx)
                row = self.db.fetchone(f"SELECT 1 FROM {self.schema}.rl_q_table WHERE state_hash=%s", [h])
                if row:
                    known += 1
            coverage = dict(total=len(current_contexts), known=known, ratio=(known/len(current_contexts) if current_contexts else 0.0))
        # Метаданные обучения
        last_train = self.last_train_at()
        last_updates = self.last_train_updates()
        return {
            'n_states': int(agg.get('n_states') or 0),
            'avg_visits': float(agg.get('avg_visits') or 0.0),
            'sum_visits': int(agg.get('sum_visits') or 0),
            'mean_abs_q': {
                '1': float(agg.get('mean_abs_q1') or 0.0),
                '2': float(agg.get('mean_abs_q2') or 0.0),
                '3': float(agg.get('mean_abs_q3') or 0.0),
            },
            'last_update': str(agg.get('last_update') or ''),
            'last_train_ts': last_train,
            'last_train_updates': last_updates,
            'coverage': coverage
        }

    @staticmethod
    def _bucket(v):
        if v is None:
            return 'NA'
        try:
            v = float(v)
        except:
            return 'NA'
        if v < 1: return 'L1'
        if v < 5: return 'L5'
        if v < 10: return 'L10'
        if v < 20: return 'L20'
        return 'LXX'

    @staticmethod
    def _year_bucket(y):
        if y is None: return 'NA'
        try:
            y = int(y)
        except:
            return 'NA'
        if y < 1950: return 'old'
        if y < 1990: return 'mid'
        if y < 2010: return 'new'
        return 'recent'

# --- Расширенный QListWidget для изображений (drop из ФС) ---
class ImageListWidget(QtWidgets.QListWidget):
    filesDropped = QtCore.pyqtSignal(list)  # список путей

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = []
            for url in e.mimeData().urls():
                p = url.toLocalFile()
                if p and Path(p).suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'):
                    paths.append(p)
            if paths:
                self.filesDropped.emit(paths)
                e.acceptProposedAction()
                return
        super().dropEvent(e)

# --- Главный виджет ---
class UrbanViewWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('UrbanView v3.3.0: Проекты, PDF, RL-графики, слой QGIS, RL метрики')
        self.resize(1500, 930)

        # Служебные
        self.settings = QtCore.QSettings('UrbanView', 'UrbanViewWidget')
        self.db = Pg(PG)
        self.rl = RLAgent(self.db, SCHEMA)
        self.wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        self.current_project_id = None
        self.current_name_project_id = None
        self.session_dataset_rows = set()
        self.projects_cache = []
        self.projectsLayer = None
        self.dirty = False

        # RL plot helpers (matplotlib)
        self.rlFig = None
        self.rlCanvas = None
        self.rlAx = None
        self._bar_to_index = {}

        # RL рекомендации для изображений (path -> pr)
        self._rl_suggestions_map = {}

        # Мини-канвас превью
        self.previewCanvas = None
        self.previewMarker = None

        # UI
        self._build_ui()
        self._load_dictionaries()
        self._load_projects_list()
        self._restore_ui_state()
        self._update_train_status()   # индикатор RL
        self._info('Готово')

    # ---------- UI построение ----------
    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)

        # Верхняя панель действий
        top = QtWidgets.QHBoxLayout()
        self.btnNew = QtWidgets.QPushButton('Новый')
        self.btnSave = QtWidgets.QPushButton('Сохранить')
        self.btnPDF = QtWidgets.QPushButton('Экспорт PDF')
        self.chkOpenPDF = QtWidgets.QCheckBox('Открыть PDF')
        self.btnDatasetAppend = QtWidgets.QPushButton('Обновить Dataset (append)')
        self.btnOpenPDFDir = QtWidgets.QPushButton('Папка PDF')
        self.btnOpenDatasetDir = QtWidgets.QPushButton('Папка Dataset')
        self.btnLoadLayer = QtWidgets.QPushButton('Загрузить слой проектов')
        self.btnZoom = QtWidgets.QPushButton('Зум к проекту')
        self.btnShowAll = QtWidgets.QPushButton('Показать все проекты')

        self.btnNew.clicked.connect(self._new_project)
        self.btnSave.clicked.connect(self._save_project)
        self.btnPDF.clicked.connect(self._export_pdf)
        self.btnDatasetAppend.clicked.connect(self._build_dataset_append)
        self.btnOpenPDFDir.clicked.connect(lambda: open_in_file_manager(PDF_OUT_DIR))
        self.btnOpenDatasetDir.clicked.connect(lambda: open_in_file_manager(DATASET_IMG_DIR))
        self.btnLoadLayer.clicked.connect(self._load_projects_layer)
        self.btnZoom.clicked.connect(self._zoom_to_project)
        self.btnShowAll.clicked.connect(self._show_all_projects)

        for w in (self.btnNew, self.btnSave, self.btnPDF, self.chkOpenPDF, self.btnDatasetAppend,
                  self.btnOpenPDFDir, self.btnOpenDatasetDir, self.btnLoadLayer, self.btnZoom, self.btnShowAll):
            top.addWidget(w)
        top.addStretch(1)
        main.addLayout(top)

        # Сплиттер: слева список проектов, справа вкладки
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter = splitter

        # Левая панель: фильтры + QTreeWidget
        leftWrap = QtWidgets.QWidget()
        leftLay = QtWidgets.QVBoxLayout(leftWrap)
        fl = QtWidgets.QHBoxLayout()
        self.edFilter = QtWidgets.QLineEdit()
        self.edFilter.setPlaceholderText('Фильтр по имени/статусу/году/типологии/ВРИ...')
        self.edFilter.textChanged.connect(self._apply_filter)
        self.edFilterTags = QtWidgets.QLineEdit()
        self.edFilterTags.setPlaceholderText('Фильтр по тэгам (через ,)')
        self.edFilterTags.textChanged.connect(self._apply_filter)
        fl.addWidget(self.edFilter); fl.addWidget(self.edFilterTags)
        self.treeProjects = QtWidgets.QTreeWidget()
        self.treeProjects.setAlternatingRowColors(True)
        self.treeProjects.setHeaderLabels(['Имя', 'Статус', 'Год', 'Тэги', 'Типология', 'ВРИ'])
        self.treeProjects.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeProjects.customContextMenuRequested.connect(self._projects_ctx_menu)
        self.treeProjects.itemDoubleClicked.connect(self._open_project_from_item)
        leftLay.addLayout(fl)
        leftLay.addWidget(self.treeProjects)
        splitter.addWidget(leftWrap)

        # Правая панель: вкладки
        rightWrap = QtWidgets.QWidget()
        rightLay = QtWidgets.QVBoxLayout(rightWrap)
        self.tabs = QtWidgets.QTabWidget()

        # Вкладка Project
        tabProject = QtWidgets.QWidget(); frmP = QtWidgets.QFormLayout(tabProject)
        self.edName = QtWidgets.QLineEdit()
        self.cbFno = QtWidgets.QComboBox()
        self.cbViewObject = QtWidgets.QComboBox()
        self.cbMorphotype = QtWidgets.QComboBox()
        self.cbTypology = QtWidgets.QComboBox()
        self.cbVri = QtWidgets.QComboBox()

        for w in (self.edName,):
            w.textChanged.connect(self._mark_dirty)
        for cb in (self.cbFno, self.cbViewObject, self.cbMorphotype, self.cbTypology, self.cbVri):
            cb.currentIndexChanged.connect(self._mark_dirty)

        # Теги
        tagBox = QtWidgets.QVBoxLayout()
        self.listTags = QtWidgets.QListWidget()
        self.listTags.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.listTags.setAlternatingRowColors(True)
        self.listTags.itemChanged.connect(self._mark_dirty)
        tagAddLay = QtWidgets.QHBoxLayout()
        self.edTagNew = QtWidgets.QLineEdit(); self.edTagNew.setPlaceholderText('Новый тэг...')
        self.btnTagAdd = QtWidgets.QPushButton('Добавить тэг')
        self.btnTagAdd.clicked.connect(self._add_tag)
        tagAddLay.addWidget(self.edTagNew); tagAddLay.addWidget(self.btnTagAdd)
        tagBox.addWidget(QtWidgets.QLabel('ТЭГИ:'))
        tagBox.addWidget(self.listTags)
        tagBox.addLayout(tagAddLay)

        # Координаты + мини-превью
        coordCol = QtWidgets.QVBoxLayout()
        coordLay = QtWidgets.QHBoxLayout()
        self.edX = QtWidgets.QLineEdit(); self.edX.setPlaceholderText('x (lon, EPSG:4326)')
        self.edY = QtWidgets.QLineEdit(); self.edY.setPlaceholderText('y (lat, EPSG:4326)')
        self.edX.textChanged.connect(self._on_coord_changed)
        self.edY.textChanged.connect(self._on_coord_changed)
        self.btnPickPoint = QtWidgets.QPushButton('Выбрать на карте')
        self.btnPickPoint.clicked.connect(self._pick_point)
        coordLay.addWidget(self.edX); coordLay.addWidget(self.edY); coordLay.addWidget(self.btnPickPoint)
        coordCol.addLayout(coordLay)
        try:
            self.previewCanvas = QgsMapCanvas()
            self.previewCanvas.setMinimumHeight(140)
            self.previewCanvas.setCrsTransformEnabled(True)
            self.previewCanvas.setDestinationCrs(self.wgs84)
            self.previewCanvas.setCanvasColor(QtGui.QColor(245, 245, 245))
            coordCol.addWidget(self.previewCanvas)
        except Exception:
            coordCol.addWidget(QtWidgets.QLabel('Превью карты недоступно'))
        self.txtSummary = QtWidgets.QTextEdit(); self.txtSummary.setReadOnly(True); self.txtSummary.setFixedHeight(100)

        frmP.addRow('Наименование:', self.edName)
        frmP.addRow('ФНО:', self.cbFno)
        frmP.addRow('Вид объекта:', self.cbViewObject)
        frmP.addRow('Морфотип:', self.cbMorphotype)
        frmP.addRow('Типология:', self.cbTypology)
        frmP.addRow('ВРИ:', self.cbVri)
        frmP.addRow('Координаты:', coordCol)
        frmP.addRow(tagBox)
        frmP.addRow(QtWidgets.QLabel('Сводка:'), self.txtSummary)

        # Вкладка TEP
        tabTep = QtWidgets.QWidget(); frmT = QtWidgets.QFormLayout(tabTep)
        self.cbStatus = QtWidgets.QComboBox()
        self.cbStatus.addItems(['', 'построен','проект','строительство','заброшен'])
        self.cbStatus.currentIndexChanged.connect(self._mark_dirty)
        self.spArea = self._dbl(); self.spArea.valueChanged.connect(self._mark_dirty)
        self.spFoot = self._dbl(); self.spFoot.valueChanged.connect(self._mark_dirty)
        self.spSPP = self._dbl(); self.spSPP.valueChanged.connect(self._mark_dirty)
        self.spYear = QtWidgets.QSpinBox(); self.spYear.setRange(1800, 2100); self.spYear.setSpecialValueText(''); self.spYear.setValue(self.spYear.minimum()); self.spYear.valueChanged.connect(self._mark_dirty)
        self.spMeanLevel = self._dbl(); self.spMeanLevel.valueChanged.connect(self._mark_dirty)
        self.spMeanHeight = self._dbl(); self.spMeanHeight.valueChanged.connect(self._mark_dirty)
        self.spDensity = self._dbl(); self.spDensity.valueChanged.connect(self._mark_dirty)
        self.edDescr = QtWidgets.QPlainTextEdit(); self.edDescr.textChanged.connect(self._mark_dirty)
        self.btnAutofill = QtWidgets.QPushButton('Автозаполнить (пересечение AGR)')
        self.btnAutofill.clicked.connect(self._autofill_from_agr)

        frmT.addRow('Статус:', self.cbStatus)
        frmT.addRow('Площадь (area):', self.spArea)
        frmT.addRow('Пятно (footprint):', self.spFoot)
        frmT.addRow('СПП (spp):', self.spSPP)
        frmT.addRow('Год ввода:', self.spYear)
        frmT.addRow('Средн. этажн.:', self.spMeanLevel)
        frmT.addRow('Средн. высота:', self.spMeanHeight)
        frmT.addRow('Плотность:', self.spDensity)
        frmT.addRow('Описание:', self.edDescr)
        frmT.addRow(self.btnAutofill)

        # Вкладка Images
        tabImg = QtWidgets.QWidget(); vImg = QtWidgets.QVBoxLayout(tabImg)
        self.listImages = ImageListWidget()
        self.listImages.setViewMode(QtWidgets.QListView.IconMode)
        self.listImages.setIconSize(QtCore.QSize(160, 120))
        self.listImages.setResizeMode(QtWidgets.QListView.Adjust)
        self.listImages.setMovement(QtWidgets.QListView.Snap)
        self.listImages.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listImages.setSpacing(8)
        self.listImages.filesDropped.connect(self._on_image_files_dropped)
        self.listImages.model().rowsMoved.connect(self._on_images_reordered)
        self.listImages.itemChanged.connect(self._mark_dirty)
        self.listImages.itemDoubleClicked.connect(self._open_image_file)

        imgBtn = QtWidgets.QHBoxLayout()
        self.btnAddImg = QtWidgets.QPushButton('Добавить изображения')
        self.btnAddImgDir = QtWidgets.QPushButton('Добавить папку')
        self.btnRmImg = QtWidgets.QPushButton('Удалить выбранные')
        imgBtn.addWidget(self.btnAddImg); imgBtn.addWidget(self.btnAddImgDir); imgBtn.addWidget(self.btnRmImg); imgBtn.addStretch(1)
        self.btnAddImg.clicked.connect(self._add_images_files)
        self.btnAddImgDir.clicked.connect(self._add_images_dir)
        self.btnRmImg.clicked.connect(self._remove_selected_images)

        imgRL = QtWidgets.QHBoxLayout()
        self.btnImgRLSuggest = QtWidgets.QPushButton('Показать рекомендации (RL)')
        self.btnImgApplyRL = QtWidgets.QPushButton('Применить рекомендации')
        self.btnImgClearRL = QtWidgets.QPushButton('Скрыть рекомендации')
        imgRL.addWidget(self.btnImgRLSuggest)
        imgRL.addWidget(self.btnImgApplyRL)
        imgRL.addWidget(self.btnImgClearRL)
        imgRL.addStretch(1)
        self.btnImgRLSuggest.clicked.connect(self._images_show_rl_suggestions)
        self.btnImgApplyRL.clicked.connect(self._images_apply_rl_suggestions)
        self.btnImgClearRL.clicked.connect(self._images_clear_rl_suggestions)

        vImg.addWidget(QtWidgets.QLabel('Перетащите из проводника или переставьте для приоритета (позиция=1..3). Максимум 3.'))
        vImg.addLayout(imgRL)
        vImg.addWidget(self.listImages)
        vImg.addLayout(imgBtn)

        # Вкладка RL
        tabRL = QtWidgets.QWidget(); vRL = QtWidgets.QVBoxLayout(tabRL)
        rlBtns = QtWidgets.QHBoxLayout()
        self.btnRLSuggest = QtWidgets.QPushButton('Предложить приоритеты (RL, изменить порядок)')
        self.btnRLTrain = QtWidgets.QPushButton('Переобучить на истории')
        self.btnRLReset = QtWidgets.QPushButton('Сбросить текущее состояние')
        self.btnRLFeedback = QtWidgets.QPushButton('Записать обратную связь (по текущему порядку)')
        self.btnRLMetrics = QtWidgets.QPushButton('Метрики RL')
        self.btnRLSuggest.clicked.connect(self._rl_suggest)
        self.btnRLTrain.clicked.connect(self._rl_train)
        self.btnRLReset.clicked.connect(self._rl_reset_state)
        self.btnRLFeedback.clicked.connect(self._rl_feedback_from_current_order)
        self.btnRLMetrics.clicked.connect(self._rl_show_metrics)
        rlBtns.addWidget(self.btnRLSuggest); rlBtns.addWidget(self.btnRLTrain); rlBtns.addWidget(self.btnRLReset); rlBtns.addWidget(self.btnRLFeedback); rlBtns.addWidget(self.btnRLMetrics); rlBtns.addStretch(1)

        # Индикатор обучения
        statLay = QtWidgets.QHBoxLayout()
        self.lblTrainMarker = QtWidgets.QLabel('●')
        self.lblTrainMarker.setToolTip('Индикатор обучения RL')
        self.lblTrainStamp = QtWidgets.QLabel('Последнее обучение: —')
        self.lblTrainMarker.setStyleSheet('font-size:18px; color:#aaa;')
        self.lblTrainStamp.setStyleSheet('color:#666;')
        statLay.addWidget(self.lblTrainMarker)
        statLay.addWidget(self.lblTrainStamp)
        statLay.addStretch(1)

        self.tblQ = QtWidgets.QTableWidget(0, 4)
        self.tblQ.setHorizontalHeaderLabels(['Изображение', 'Q(a=1)', 'Q(a=2)', 'Q(a=3)'])
        self.tblQ.horizontalHeader().setStretchLastSection(True)

        # Матплотлиб-диаграмма
        if HAS_MPL:
            self.rlFig = Figure(figsize=(6, 3), constrained_layout=True)
            self.rlCanvas = FigureCanvas(self.rlFig)
            self.rlAx = self.rlFig.add_subplot(111)
            vRL.addLayout(rlBtns)
            vRL.addLayout(statLay)
            vRL.addWidget(QtWidgets.QLabel('Q-значения (таблица):'))
            vRL.addWidget(self.tblQ, stretch=1)
            vRL.addWidget(QtWidgets.QLabel('Q-значения (диаграмма, matplotlib):'))
            vRL.addWidget(self.rlCanvas, stretch=2)
            self.rlCanvas.mpl_connect('pick_event', self._on_bar_pick)
        else:
            vRL.addLayout(rlBtns)
            vRL.addLayout(statLay)
            vRL.addWidget(QtWidgets.QLabel('Q-значения (таблица):'))
            vRL.addWidget(self.tblQ, stretch=1)
            lbl = QtWidgets.QLabel(f'График недоступен: matplotlib не найден ({MPL_IMPORT_ERROR}).')
            lbl.setStyleSheet('color:#a00')
            vRL.addWidget(lbl)

        # Вкладка Tools
        tabTools = QtWidgets.QWidget(); vTools = QtWidgets.QVBoxLayout(tabTools)
        self.btnDatasetFull = QtWidgets.QPushButton('Регенерировать CSV датасета (полностью)')
        self.btnDatasetFull.clicked.connect(self._build_dataset_full)
        self.btnLoadLayer2 = QtWidgets.QPushButton('Загрузить/обновить слой проектов в QGIS')
        self.btnLoadLayer2.clicked.connect(self._load_projects_layer)
        self.btnExportQ = QtWidgets.QPushButton('Экспорт Q-таблицы в JSON')
        self.btnImportQ = QtWidgets.QPushButton('Импорт Q-таблицы из JSON')
        self.btnResetQTable = QtWidgets.QPushButton('Полный сброс Q-таблицы (ОСТОРОЖНО)')
        self.btnExportQ.clicked.connect(self._export_q_table_json)
        self.btnImportQ.clicked.connect(self._import_q_table_json)
        self.btnResetQTable.clicked.connect(self._reset_q_table)
        vTools.addWidget(self.btnDatasetFull)
        vTools.addWidget(self.btnLoadLayer2)
        vTools.addWidget(self.btnExportQ)
        vTools.addWidget(self.btnImportQ)
        vTools.addWidget(self.btnResetQTable)
        vTools.addStretch(1)

        self.tabs.addTab(tabProject, 'Project')
        self.tabs.addTab(tabTep, 'TEP')
        self.tabs.addTab(tabImg, 'Images')
        self.tabs.addTab(tabRL, 'RL')
        self.tabs.addTab(tabTools, 'Tools')
        rightLay.addWidget(self.tabs)

        splitter.addWidget(rightWrap)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter)

        # Нижняя статус-строка
        self.status = QtWidgets.QLabel()
        main.addWidget(self.status)

    def _dbl(self):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(-1e9, 1e9)
        sp.setDecimals(4)
        sp.setSingleStep(0.1)
        sp.setSpecialValueText('')
        return sp

    # ---------- Сохранение/восстановление UI ----------
    def _restore_ui_state(self):
        if self.settings.value('geometry'):
            self.restoreGeometry(self.settings.value('geometry', type=QtCore.QByteArray))
        if self.settings.value('splitter'):
            self.splitter.restoreState(self.settings.value('splitter', type=QtCore.QByteArray))

    def closeEvent(self, e):
        try:
            if not self._maybe_save():
                e.ignore()
                return
            self.settings.setValue('geometry', self.saveGeometry())
            self.settings.setValue('splitter', self.splitter.saveState())
            self.db.close()
        except:
            pass
        super().closeEvent(e)

    # ---------- Unsaved tracking ----------
    def _mark_dirty(self, *args, **kwargs):
        self.dirty = True
        self._update_summary()

    def _clear_dirty(self):
        self.dirty = False

    def _maybe_save(self):
        if not self.dirty:
            return True
        ret = QtWidgets.QMessageBox.question(self, 'Сохранить изменения?', 'Есть несохраненные изменения. Сохранить?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
        if ret == QtWidgets.QMessageBox.Yes:
            self._save_project()
            return True
        if ret == QtWidgets.QMessageBox.No:
            return True
        return False

    # ---------- Проекты: загрузка, фильтр, открытие ----------
    def _load_projects_list(self):
        self.treeProjects.clear()
        rows = self.db.fetchall(f"""
            SELECT pc.id, np.name as name_project, pc.status, pc.year_entry,
                   (
                       SELECT string_agg(t.name, ', ' ORDER BY t.name)
                       FROM {SCHEMA}.project_teg pt
                       JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
                       WHERE pt.project_id=pc.id
                   ) AS tags,
                   ty.name as typology,
                   vr.name as vri
            FROM {SCHEMA}.project_card pc
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            LEFT JOIN {SCHEMA}.typology ty ON ty.id=pc.id_typology
            LEFT JOIN {SCHEMA}.vri vr ON vr.id=pc.vri_id
            ORDER BY LOWER(np.name)
        """)
        self.projects_cache = rows
        for r in rows:
            it = QtWidgets.QTreeWidgetItem([
                r['name_project'] or f'ID {r["id"]}',
                r['status'] or '',
                '' if r['year_entry'] is None else str(r['year_entry']),
                r['tags'] or '',
                r['typology'] or '',
                r['vri'] or ''
            ])
            it.setData(0, QtCore.Qt.UserRole, r['id'])
            self.treeProjects.addTopLevelItem(it)
        self.treeProjects.resizeColumnToContents(1)

    def _apply_filter(self):
        text = (self.edFilter.text() or '').lower().strip()
        tagtext = (self.edFilterTags.text() or '').lower().strip()
        tag_tokens = [t.strip() for t in tagtext.split(',') if t.strip()]
        self.treeProjects.clear()
        for r in self.projects_cache:
            s = ' '.join([
                (r['name_project'] or ''),
                (r.get('status') or ''),
                '' if r.get('year_entry') is None else str(r.get('year_entry')),
                (r.get('typology') or ''),
                (r.get('vri') or '')
            ]).lower()
            tags = (r.get('tags') or '').lower()
            if text and text not in s:
                continue
            if tag_tokens and not all(tok in tags for tok in tag_tokens):
                continue
            it = QtWidgets.QTreeWidgetItem([
                r['name_project'] or f'ID {r["id"]}',
                r.get('status') or '',
                '' if r.get('year_entry') is None else str(r.get('year_entry')),
                r.get('tags') or '',
                r.get('typology') or '',
                r.get('vri') or ''
            ])
            it.setData(0, QtCore.Qt.UserRole, r['id'])
            self.treeProjects.addTopLevelItem(it)

    def _projects_ctx_menu(self, pos):
        it = self.treeProjects.itemAt(pos)
        if not it:
            return
        pid = it.data(0, QtCore.Qt.UserRole)
        menu = QtWidgets.QMenu(self)
        aOpen = menu.addAction('Открыть')
        aDup = menu.addAction('Дублировать')
        aPdf = menu.addAction('Экспорт PDF')
        aDel = menu.addAction('Удалить')
        act = menu.exec_(self.treeProjects.mapToGlobal(pos))
        if act == aOpen:
            self._open_project(pid)
        elif act == aDup:
            self._duplicate_project(pid)
        elif act == aPdf:
            self._open_project(pid)
            self._export_pdf()
        elif act == aDel:
            self._delete_project(pid)

    def _open_project_from_item(self, item, col):
        pid = item.data(0, QtCore.Qt.UserRole)
        self._open_project(pid)

    def _open_project(self, pid):
        if not self._maybe_save():
            return
        r = self.db.fetchone(f"""
            SELECT pc.*,
                   np.name as name_project
            FROM {SCHEMA}.project_card pc
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            WHERE pc.id=%s
        """, [pid])
        if not r:
            return
        self.current_project_id = pid
        self.current_name_project_id = r['id_name_project']

        self.edName.setText(r['name_project'] or '')
        self._set_combo_by_id(self.cbFno, r['id_fno_group'])
        self._set_combo_by_id(self.cbViewObject, r['id_view_object'])
        self._set_combo_by_id(self.cbMorphotype, r['id_morphotype'])
        self._set_combo_by_id(self.cbTypology, r['id_typology'])
        self._set_combo_by_id(self.cbVri, r['vri_id'])

        self.edX.setText('' if r['x'] is None else str(r['x']))
        self.edY.setText('' if r['y'] is None else str(r['y']))
        self.cbStatus.setCurrentText(r['status'] or '')
        self._set_spin(self.spArea, r['area'])
        self._set_spin(self.spFoot, r['footprint'])
        self._set_spin(self.spSPP, r['spp'])
        self._set_spin(self.spMeanLevel, r['mean_level'])
        self._set_spin(self.spMeanHeight, r['mean_height'])
        self._set_spin(self.spDensity, r['density'])
        if r['year_entry'] is not None:
            self.spYear.setValue(int(r['year_entry']))
        else:
            self.spYear.setValue(self.spYear.minimum())
        self.edDescr.setPlainText(r['description'] or '')

        # Теги
        tags = self.db.fetchall(f"""
            SELECT t.name
            FROM {SCHEMA}.project_teg pt
            JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
            WHERE pt.project_id=%s
        """, [pid])
        self._set_tags_checked(set([x['name'] for x in tags]))

        # Изображения
        self.listImages.clear()
        self._rl_suggestions_map.clear()
        imgs = self.db.fetchall(f"""
            SELECT path, priority FROM {SCHEMA}.project_image
            WHERE project_id=%s
            ORDER BY priority
        """, [pid])
        for im in imgs:
            self._add_image_item(im['path'])

        self._update_summary()
        self._update_preview_marker()
        self._clear_dirty()
        self._update_train_status()
        self._info(f'Открыт проект ID={pid}')

    def _new_project(self):
        if not self._maybe_save():
            return
        self.current_project_id = None
        self.current_name_project_id = None
        self.edName.clear()
        for cb in (self.cbFno, self.cbViewObject, self.cbMorphotype, self.cbTypology, self.cbVri):
            cb.setCurrentIndex(0)
        for w in (self.edX, self.edY):
            w.clear()
        self.cbStatus.setCurrentIndex(0)
        for sp in (self.spArea, self.spFoot, self.spSPP, self.spMeanLevel, self.spMeanHeight, self.spDensity):
            sp.setValue(sp.minimum())
        self.spYear.setValue(self.spYear.minimum())
        self.edDescr.clear()
        for i in range(self.listTags.count()):
            self.listTags.item(i).setCheckState(QtCore.Qt.Unchecked)
        self.listImages.clear()
        self._rl_suggestions_map.clear()
        self._update_summary()
        self._update_preview_marker()
        self._clear_dirty()
        self._update_train_status()
        self._info('Новая карточка')

    def _duplicate_project(self, pid):
        if not self._maybe_save():
            return
        r = self.db.fetchone(f"""
            SELECT pc.*, np.name as name_project
            FROM {SCHEMA}.project_card pc
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            WHERE pc.id=%s
        """, [pid])
        if not r:
            return
        new_name = f"{r['name_project'] or 'Без имени'} (копия)"
        self.edName.setText(new_name)
        self._set_combo_by_id(self.cbFno, r['id_fno_group'])
        self._set_combo_by_id(self.cbViewObject, r['id_view_object'])
        self._set_combo_by_id(self.cbMorphotype, r['id_morphotype'])
        self._set_combo_by_id(self.cbTypology, r['id_typology'])
        self._set_combo_by_id(self.cbVri, r['vri_id'])
        self.edX.setText('' if r['x'] is None else str(r['x']))
        self.edY.setText('' if r['y'] is None else str(r['y']))
        self.cbStatus.setCurrentText(r['status'] or '')
        self._set_spin(self.spArea, r['area'])
        self._set_spin(self.spFoot, r['footprint'])
        self._set_spin(self.spSPP, r['spp'])
        self._set_spin(self.spMeanLevel, r['mean_level'])
        self._set_spin(self.spMeanHeight, r['mean_height'])
        self._set_spin(self.spDensity, r['density'])
        if r['year_entry'] is not None:
            self.spYear.setValue(int(r['year_entry']))
        else:
            self.spYear.setValue(self.spYear.minimum())
        self.edDescr.setPlainText(r['description'] or '')

        tags = self.db.fetchall(f"""
            SELECT t.name
            FROM {SCHEMA}.project_teg pt
            JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
            WHERE pt.project_id=%s
        """, [pid])
        self._set_tags_checked(set([x['name'] for x in tags]))

        self.listImages.clear()
        self._rl_suggestions_map.clear()
        imgs = self.db.fetchall(f"""
            SELECT path, priority FROM {SCHEMA}.project_image
            WHERE project_id=%s
            ORDER BY priority
        """, [pid])
        for im in imgs:
            self._add_image_item(im['path'])

        self.current_project_id = None
        self.current_name_project_id = None
        self._update_summary()
        self._update_preview_marker()
        self._clear_dirty()
        self._update_train_status()
        self._info('Данные скопированы в новую карточку. Сохраните, чтобы создать проект.')

    def _delete_project(self, pid):
        if QtWidgets.QMessageBox.question(self, 'Удаление', f'Удалить проект ID={pid}?') != QtWidgets.QMessageBox.Yes:
            return
        self.db.execute(f"DELETE FROM {SCHEMA}.project_teg WHERE project_id=%s", [pid], commit=True)
        self.db.execute(f"DELETE FROM {SCHEMA}.project_image WHERE project_id=%s", [pid], commit=True)
        self.db.execute(f"DELETE FROM {SCHEMA}.tep_project WHERE id_project=%s", [pid], commit=True)
        self.db.execute(f"DELETE FROM {SCHEMA}.project_card WHERE id=%s", [pid], commit=True)
        self._load_projects_list()
        if self.current_project_id == pid:
            self._new_project()
        self._info(f'Проект ID={pid} удален.')

    # ---------- Справочники и теги ----------
    def _load_dictionaries(self):
        def fill_combo(cb, rows):
            cb.clear()
            cb.addItem('', None)
            for r in rows:
                cb.addItem(r['name'], r['id'])
        self.dict_fno = self.db.fetchall(f"SELECT id,name FROM {SCHEMA}.fno_group ORDER BY name")
        self.dict_vo = self.db.fetchall(f"SELECT id,name FROM {SCHEMA}.view_object ORDER BY name")
        self.dict_mt = self.db.fetchall(f"SELECT id,name FROM {SCHEMA}.morphotype ORDER BY name")
        self.dict_ty = self.db.fetchall(f"SELECT id,name FROM {SCHEMA}.typology ORDER BY name")
        self.dict_vri = self.db.fetchall(f"SELECT id,name FROM {SCHEMA}.vri ORDER BY name")

        fill_combo(self.cbFno, self.dict_fno)
        fill_combo(self.cbViewObject, self.dict_vo)
        fill_combo(self.cbMorphotype, self.dict_mt)
        fill_combo(self.cbTypology, self.dict_ty)
        fill_combo(self.cbVri, self.dict_vri)

        self.listTags.clear()
        tags = self.db.fetchall(f" SELECT id, name FROM {SCHEMA}.teg ORDER BY name ")
        self.tags_dict = {r['name']: r['id'] for r in tags}
        for r in tags:
            it = QtWidgets.QListWidgetItem(r['name'])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.listTags.addItem(it)

    def _add_tag(self):
        name = (self.edTagNew.text() or '').strip()
        if not name:
            return
        row = self.db.fetchone(f"SELECT id FROM {SCHEMA}.teg WHERE name=%s", [name])
        if not row:
            self.db.execute(f"INSERT INTO {SCHEMA}.teg(name) VALUES (%s)", [name], commit=True)
        self.edTagNew.clear()
        self._load_dictionaries()
        self._mark_dirty()
        self._info(f'Тэг добавлен: {name}')

    def _set_tags_checked(self, tagset):
        self.listTags.blockSignals(True)
        for i in range(self.listTags.count()):
            it = self.listTags.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.text() in tagset else QtCore.Qt.Unchecked)
        self.listTags.blockSignals(False)

    def _collect_tags_selected(self):
        res = []
        for i in range(self.listTags.count()):
            it = self.listTags.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                res.append(it.text())
        return res

    # ---------- Координаты ----------
    def _pick_point(self):
        self.mapTool = QgsMapToolEmitPoint(self._iface().mapCanvas())
        self.mapTool.canvasClicked.connect(self._on_canvas_click)
        self._iface().mapCanvas().setMapTool(self.mapTool)
        QtWidgets.QMessageBox.information(self, 'Выбор точки', 'Кликните на карте для выбора точки (WGS84).')

    def _iface(self):
        import qgis.utils
        return qgis.utils.iface

    def _on_canvas_click(self, pt, btn):
        srcCrs = self._iface().mapCanvas().mapSettings().destinationCrs()
        if srcCrs != self.wgs84:
            xform = QgsCoordinateTransform(srcCrs, self.wgs84, QgsProject.instance())
            pt = xform.transform(pt)
        self.edX.setText(f'{pt.x():.8f}')
        self.edY.setText(f'{pt.y():.8f}')
        self._mark_dirty()
        self._info(f'Точка: x={pt.x():.6f}, y={pt.y():.6f}')

    def _zoom_to_project(self):
        x = self._get_float(self.edX.text())
        y = self._get_float(self.edY.text())
        if x is None or y is None:
            QtWidgets.QMessageBox.information(self, 'Зум', 'Нет корректных координат x,y.')
            return
        canvas = self._iface().mapCanvas()
        dx = 0.003
        rect = QgsRectangle(x - dx, y - dx, x + dx, y + dx)
        if canvas.mapSettings().destinationCrs() != self.wgs84:
            xform = QgsCoordinateTransform(self.wgs84, canvas.mapSettings().destinationCrs(), QgsProject.instance())
            rect = xform.transform(rect)
        canvas.setExtent(rect)
        canvas.refresh()

    def _show_all_projects(self):
        self._load_projects_layer()
        if not (self.projectsLayer and self.projectsLayer.isValid()):
            return
        canvas = self._iface().mapCanvas()
        if self.projectsLayer.featureCount() > 0:
            canvas.setExtent(self.projectsLayer.extent())
            canvas.refresh()
            self._info('Показаны все проекты.')
        else:
            self._info('В слое нет объектов.')

    # ---------- AGR автозаполнение ----------
    def _autofill_from_agr(self):
        x = self._get_float(self.edX.text())
        y = self._get_float(self.edY.text())
        if x is None or y is None:
            QtWidgets.QMessageBox.warning(self, 'Координата', 'Укажите корректные x,y (WGS84).')
            return
        r = self.db.fetchone(f"""
            SELECT id, name, area, year, spp_all, spp_live, spp_unlive
            FROM {SCHEMA}.agr
            WHERE ST_Intersects(geom, ST_SetSRID(ST_Point(%s,%s),4326))
            ORDER BY id
            LIMIT 1
        """, [x, y])
        if not r:
            self._info('Пересечений AGR не найдено.')
            return
        self.spArea.setValue(self._nz(r['area']))
        if r['year'] is not None:
            self.spYear.setValue(int(r['year']))
        self.spSPP.setValue(self._nz(r['spp_all']))
        self._mark_dirty()
        self._info('ТЭП подставлены из AGR. Сохраните проект.')

    # ---------- Изображения ----------
    def _on_image_files_dropped(self, paths):
        if self.listImages.count() >= 3:
            QtWidgets.QMessageBox.warning(self, 'Изображения', 'Максимум 3 изображения.')
            return
        for p in paths:
            if self.listImages.count() >= 3:
                break
            self._add_image_item(p)
        self._images_clear_rl_suggestions()
        self._mark_dirty()

    def _add_images_files(self):
        if self.listImages.count() >= 3:
            QtWidgets.QMessageBox.warning(self, 'Изображения', 'Максимум 3 изображения.')
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Выберите изображения', '', 'Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif)')
        for p in paths:
            if self.listImages.count() >= 3:
                break
            self._add_image_item(p)
        if paths:
            self._images_clear_rl_suggestions()
            self._mark_dirty()

    def _add_images_dir(self):
        if self.listImages.count() >= 3:
            QtWidgets.QMessageBox.warning(self, 'Изображения', 'Максимум 3 изображения.')
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями', '')
        if not d:
            return
        added = 0
        for p in sorted(Path(d).glob('*')):
            if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.gif'):
                self._add_image_item(str(p))
                added += 1
                if self.listImages.count() >= 3:
                    break
        if added:
            self._images_clear_rl_suggestions()
            self._mark_dirty()
            self._info(f'Добавлено изображений: {added}')

    def _add_image_item(self, path):
        if not path:
            return
        img = QtGui.QImage(path)
        if img.isNull():
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        else:
            pm = QtGui.QPixmap.fromImage(img).scaled(self.listImages.iconSize(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            icon = QtGui.QIcon(pm)
        base_text = Path(path).name
        it = QtWidgets.QListWidgetItem(icon, base_text)
        it.setToolTip(path)
        it.setData(QtCore.Qt.UserRole, path)             # полный путь
        it.setData(QtCore.Qt.UserRole + 1, base_text)    # базовый текст без RL
        pr = self._rl_suggestions_map.get(path)
        if pr:
            self._apply_item_rl_style(it, pr)
        self.listImages.addItem(it)

    def _remove_selected_images(self):
        for it in self.listImages.selectedItems():
            row = self.listImages.row(it)
            self.listImages.takeItem(row)
        self._images_clear_rl_suggestions()
        self._mark_dirty()

    def _open_image_file(self, item):
        p = item.data(QtCore.Qt.UserRole)
        if p: open_in_file_manager(p)

    def _on_images_reordered(self, *args, **kwargs):
        self._images_clear_rl_suggestions()
        self._mark_dirty()

    # ---------- RL в Images (визуальные рекомендации) ----------
    def _images_show_rl_suggestions(self):
        feats = self._current_image_features()
        if not feats:
            QtWidgets.QMessageBox.warning(self, 'RL', 'Добавьте изображения.')
            return
        ctx = self._build_project_context()
        priorities = self.rl.suggest_priorities_for_images(ctx, feats)
        self._rl_suggestions_map.clear()
        for i in range(self.listImages.count()):
            it = self.listImages.item(i)
            path = it.data(QtCore.Qt.UserRole)
            pr = priorities[i]
            self._rl_suggestions_map[path] = pr
            self._apply_item_rl_style(it, pr)
        self._info('Показаны рекомендации RL (не применены).')

    def _images_apply_rl_suggestions(self):
        if not self._rl_suggestions_map:
            self._images_show_rl_suggestions()
            if not self._rl_suggestions_map:
                return
        items = [self.listImages.item(i) for i in range(self.listImages.count())]
        items_sorted = sorted(items, key=lambda it: self._rl_suggestions_map.get(it.data(QtCore.Qt.UserRole), 99))
        paths = [it.data(QtCore.Qt.UserRole) for it in items_sorted]
        self.listImages.clear()
        for p in paths:
            self._add_image_item(p)
        self._info('Рекомендации RL применены: порядок изменён.')

    def _images_clear_rl_suggestions(self):
        self._rl_suggestions_map.clear()
        for i in range(self.listImages.count()):
            it = self.listImages.item(i)
            it.setText(self._base_item_text(it))
            it.setBackground(QtGui.QBrush())
            it.setForeground(QtGui.QBrush())

    def _apply_item_rl_style(self, item: QtWidgets.QListWidgetItem, pr: int):
        base = self._base_item_text(item)
        item.setText(f'[RL:{pr}] {base}')
        if pr == 1:
            bg = QtGui.QColor(210, 245, 210)
            fg = QtGui.QColor(0, 80, 0)
        elif pr == 2:
            bg = QtGui.QColor(255, 250, 205)
            fg = QtGui.QColor(120, 90, 0)
        else:
            bg = QtGui.QColor(235, 235, 235)
            fg = QtGui.QColor(60, 60, 60)
        item.setBackground(QtGui.QBrush(bg))
        item.setForeground(QtGui.QBrush(fg))

    def _base_item_text(self, item):
        return item.data(QtCore.Qt.UserRole + 1) or item.text()

    # ---------- RL ----------
    def _rl_suggest(self):
        ctx = self._build_project_context()
        feats = self._current_image_features()
        if not feats:
            QtWidgets.QMessageBox.warning(self, 'RL', 'Добавьте изображения.')
            return
        priorities = self.rl.suggest_priorities_for_images(ctx, feats)
        order = [i for i,_ in sorted(enumerate(priorities), key=lambda x: x[1])]
        items = [self.listImages.item(i) for i in range(self.listImages.count())]
        paths_in_order = [items[i].data(QtCore.Qt.UserRole) for i in order]
        self.listImages.clear()
        for p in paths_in_order:
            self._add_image_item(p)
        self._images_show_rl_suggestions()
        self._fill_q_table_preview()
        self._info('Приоритеты предложены (порядок изменен).')

    def _rl_train(self):
        m = self.rl.train_from_history()
        self._fill_q_table_preview()
        self._update_train_status()
        QtWidgets.QMessageBox.information(self, 'RL', f"Обучение завершено.\nОбновлений: {m['updates']}\nУникальных состояний: {m['unique_states']}\nСредняя ΔQ: {m['avg_delta']:.4f}")

    def _rl_reset_state(self):
        ctx = self._build_project_context()
        feats = self._current_image_features()
        if not feats:
            QtWidgets.QMessageBox.information(self, 'RL', 'Нет изображений для сброса состояния.')
            return
        for f in feats:
            self.rl.reset_state(dict(ctx, **f))
        self._fill_q_table_preview()
        self._images_clear_rl_suggestions()
        self._info('Состояние (Q) для текущего контекста/изображений сброшено.')

    def _rl_feedback_from_current_order(self):
        ctx = self._build_project_context()
        feats = self._current_image_features()
        if not feats:
            QtWidgets.QMessageBox.warning(self, 'RL', 'Добавьте изображения.')
            return
        rewards = {1: 1.0, 2: 0.5, 3: 0.33}
        sum_delta = 0.0
        for idx, f in enumerate(feats, start=1):
            pr = min(idx, 3)
            _, d = self.rl.update(dict(ctx, **f), pr, rewards[pr])
            sum_delta += d
        self.db.commit()
        self.rl._meta_set('last_train_ts', datetime.now().isoformat(timespec='seconds'))
        self.rl._meta_set('last_train_updates', len(feats))
        self._fill_q_table_preview()
        self._update_train_status()
        self._info('Обратная связь записана в Q-таблицу.')

    def _fill_q_table_preview(self):
        ctx = self._build_project_context()
        feats = self._current_image_features()
        self.tblQ.setRowCount(0)
        labels = []
        q_list = []
        for idx, f in enumerate(feats):
            qv, vc, h = self.rl.get_q_values(dict(ctx, **f))
            r = self.tblQ.rowCount()
            self.tblQ.insertRow(r)
            ar_tag = f.get("ar_bucket","")
            lbl = f'img#{idx+1} {ar_tag}'
            labels.append(lbl)
            self.tblQ.setItem(r, 0, QtWidgets.QTableWidgetItem(lbl))
            self.tblQ.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{float(qv.get("1",0.0)):.3f}'))
            self.tblQ.setItem(r, 2, QtWidgets.QTableWidgetItem(f'{float(qv.get("2",0.0)):.3f}'))
            self.tblQ.setItem(r, 3, QtWidgets.QTableWidgetItem(f'{float(qv.get("3",0.0)):.3f}'))
            q_list.append({'1': float(qv.get('1',0.0)), '2': float(qv.get('2',0.0)), '3': float(qv.get('3',0.0))})
        self._update_rl_chart(labels, q_list)

    def _update_train_status(self):
        # Обновляем маркер и дату обучения
        last = self.rl.last_train_at()
        self.lblTrainStamp.setText(f"Последнее обучение: {last or '—'}")
        metrics = self.rl.get_metrics()
        n_states = metrics.get('n_states', 0)
        # Зеленый если состояний > 50 и средний |Q| заметен, иначе жёлтый/серый
        mean_abs_q = metrics.get('mean_abs_q', {})
        mean_q = max(mean_abs_q.get('1',0), mean_abs_q.get('2',0), mean_abs_q.get('3',0))
        color = '#3cba54' if (n_states >= 50 and mean_q >= 0.05) else ('#f4c20d' if n_states > 0 else '#aaaaaa')
        self.lblTrainMarker.setStyleSheet(f'font-size:18px; color:{color};')

    def _rl_show_metrics(self):
        ctx = self._build_project_context()
        feats = self._current_image_features()
        full_ctxs = [dict(ctx, **f) for f in feats]
        m = self.rl.get_metrics(current_contexts=full_ctxs if feats else None)
        cov = m.get('coverage') or {'total':0,'known':0,'ratio':0}
        txt = (
            f"Состояний в Q-таблице: {m['n_states']}\n"
            f"Среднее посещений на состояние: {m['avg_visits']:.2f}\n"
            f"Суммарно посещений: {m['sum_visits']}\n"
            f"Средние |Q|: a1={m['mean_abs_q']['1']:.4f}, a2={m['mean_abs_q']['2']:.4f}, a3={m['mean_abs_q']['3']:.4f}\n"
            f"Последнее обновление Q-таблицы: {m['last_update'] or '—'}\n"
            f"Последнее обучение (мета): {m['last_train_ts'] or '—'}; обновлений: {m['last_train_updates'] or '—'}\n"
        )
        if feats:
            txt += f"Покрытие текущих изображений: {cov['known']}/{cov['total']} ({cov['ratio']*100:.1f}%)\n"
        QtWidgets.QMessageBox.information(self, 'Метрики RL', txt)

    def _update_rl_chart(self, labels, q_list):
        if not HAS_MPL:
            return
        self.rlAx.clear()
        self._bar_to_index.clear()
        n = len(labels)
        if n == 0:
            self.rlCanvas.draw_idle()
            return
        x = np.arange(n)
        width = 0.25
        q1 = [v.get('1', 0.0) for v in q_list]
        q2 = [v.get('2', 0.0) for v in q_list]
        q3 = [v.get('3', 0.0) for v in q_list]
        bars1 = self.rlAx.bar(x - width, q1, width, label='a=1', color=(80/255,160/255,1.0))
        bars2 = self.rlAx.bar(x,         q2, width, label='a=2', color=(1.0,160/255,80/255))
        bars3 = self.rlAx.bar(x + width, q3, width, label='a=3', color=(120/255,200/255,120/255))
        for idx, rect in enumerate(bars1):
            rect.set_picker(True); self._bar_to_index[rect] = idx
        for idx, rect in enumerate(bars2):
            rect.set_picker(True); self._bar_to_index[rect] = idx
        for idx, rect in enumerate(bars3):
            rect.set_picker(True); self._bar_to_index[rect] = idx
        self.rlAx.set_xticks(x)
        self.rlAx.set_xticklabels(labels, rotation=0, fontsize=9)
        ymax = max([max(q1 or [0]), max(q2 or [0]), max(q3 or [0]), 1.0])
        self.rlAx.set_ylim(0, ymax * 1.05)
        self.rlAx.set_ylabel('Q')
        self.rlAx.set_title('Q-значения по действиям (a=1..3) для изображений')
        self.rlAx.legend(loc='upper right')
        self.rlAx.grid(axis='y', alpha=0.2)
        self.rlCanvas.draw_idle()

    def _on_bar_pick(self, event):
        artist = getattr(event, 'artist', None)
        if artist is None:
            return
        idx = self._bar_to_index.get(artist)
        if idx is not None:
            self._select_image_by_index(idx)

    def _select_image_by_index(self, idx):
        if 0 <= idx < self.listImages.count():
            self.listImages.setCurrentRow(idx)

    def _current_image_features(self):
        feats = []
        for i in range(self.listImages.count()):
            p = self.listImages.item(i).data(QtCore.Qt.UserRole)
            qimg = QtGui.QImage(p)
            w, h = qimg.width(), qimg.height()
            if w<=0 or h<=0:
                b = 'unk'
            else:
                ar = w/float(h)
                b = 'tall' if ar<0.8 else ('wide' if ar>1.4 else 'squareish')
            feats.append({'ar_bucket': b})
        return feats

    def _build_project_context(self):
        # ВАЖНО: состояние должно совпадать с train_from_history (никаких name/tags)
        return {
            'fno_group': self.cbFno.currentText() or '',
            'view_object': self.cbViewObject.currentText() or '',
            'morphotype': self.cbMorphotype.currentText() or '',
            'typology': self.cbTypology.currentText() or '',
            'vri': self.cbVri.currentText() or '',
            'status': self.cbStatus.currentText() or '',
            'area_b': RLAgent._bucket(self._spin_val(self.spArea)),
            'spp_b': RLAgent._bucket(self._spin_val(self.spSPP)),
            'mean_level_b': RLAgent._bucket(self._spin_val(self.spMeanLevel)),
            'mean_height_b': RLAgent._bucket(self._spin_val(self.spMeanHeight)),
            'density_b': RLAgent._bucket(self._spin_val(self.spDensity)),
            'year_b': RLAgent._year_bucket(self._spin_val(self.spYear, intv=True))
        }

    # ---------- Сохранение проекта ----------
    def _save_project(self):
        name = (self.edName.text() or '').strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, 'Имя проекта', 'Заполните наименование.')
            return

        row = self.db.fetchone(f"SELECT id FROM {SCHEMA}.name_project WHERE name=%s", [name])
        if not row:
            self.db.execute(f"INSERT INTO {SCHEMA}.name_project(name) VALUES (%s)", [name], commit=True)
            row = self.db.fetchone(f"SELECT id FROM {SCHEMA}.name_project WHERE name=%s", [name])
        id_name_project = row['id']
        self.current_name_project_id = id_name_project

        id_fno = self.cbFno.currentData()
        id_vo = self.cbViewObject.currentData()
        id_mt = self.cbMorphotype.currentData()
        id_ty = self.cbTypology.currentData()
        id_vri = self.cbVri.currentData()

        x = self._get_float(self.edX.text())
        y = self._get_float(self.edY.text())

        status = self.cbStatus.currentText() or None
        area = self._val_or_null(self.spArea)
        foot = self._val_or_null(self.spFoot)
        spp = self._val_or_null(self.spSPP)
        year = self._val_or_null(self.spYear, intv=True)
        mean_level = self._val_or_null(self.spMeanLevel)
        mean_height = self._val_or_null(self.spMeanHeight)
        density = self._val_or_null(self.spDensity)
        descr = self.edDescr.toPlainText() or None

        pc = self.db.fetchone(f"SELECT id FROM {SCHEMA}.project_card WHERE id_name_project=%s", [id_name_project])
        if pc:
            pid = pc['id']
            self.db.execute(f"""
                UPDATE {SCHEMA}.project_card
                SET id_fno_group=%s, id_view_object=%s, id_morphotype=%s, id_typology=%s,
                    id_name_project=%s, vri_id=%s,
                    x=%s, y=%s, status=%s, area=%s, footprint=%s, spp=%s,
                    year_entry=%s, mean_level=%s, mean_height=%s, density=%s,
                    description=%s,
                    geom = CASE WHEN %s IS NOT NULL THEN ST_SetSRID(ST_Point(%s,%s),4326) ELSE geom END
                WHERE id=%s
            """, [id_fno, id_vo, id_mt, id_ty, id_name_project, id_vri,
                  x, y, status, area, foot, spp, year, mean_level, mean_height, density, descr,
                  x if (x is not None and y is not None) else None, x, y, pid], commit=True)
            self.current_project_id = pid
        else:
            res = self.db.execute(f"""
                INSERT INTO {SCHEMA}.project_card
                (id_fno_group,id_view_object,id_morphotype,id_typology,id_name_project,vri_id,
                 x,y,status,area,footprint,spp,year_entry,mean_level,mean_height,density,description,geom)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        CASE WHEN %s IS NOT NULL THEN ST_SetSRID(ST_Point(%s,%s),4326) ELSE NULL END)
                RETURNING id
            """, [id_fno, id_vo, id_mt, id_ty, id_name_project, id_vri,
                  x, y, status, area, foot, spp, year, mean_level, mean_height, density, descr,
                  x if (x is not None and y is not None) else None, x, y], commit=True, returning=True)
            self.current_project_id = res['id'] if res else None

        self._sync_project_tags(self.current_project_id, self._collect_tags_selected())
        self._save_images_for_project()

        if x is not None and y is not None:
            self._save_tep_from_agr_if_any(self.current_project_id, id_name_project, x, y)

        self._load_projects_list()
        self._update_summary()
        self._clear_dirty()
        self._info(f'Сохранено. ID={self.current_project_id}')

    def _sync_project_tags(self, project_id, tag_names):
        self.db.execute(f"DELETE FROM {SCHEMA}.project_teg WHERE project_id=%s", [project_id], commit=True)
        for tname in tag_names:
            row = self.db.fetchone(f"SELECT id FROM {SCHEMA}.teg WHERE name=%s", [tname])
            if not row:
                self.db.execute(f"INSERT INTO {SCHEMA}.teg(name) VALUES (%s)", [tname], commit=True)
                row = self.db.fetchone(f"SELECT id FROM {SCHEMA}.teg WHERE name=%s", [tname])
            self.db.execute(f"""
                INSERT INTO {SCHEMA}.project_teg(project_id, teg_id) VALUES (%s,%s)
                ON CONFLICT DO NOTHING
            """, [project_id, row['id']], commit=True)

    def _save_images_for_project(self):
        if not self.current_project_id:
            return
        row = self.db.fetchone(f"""
            SELECT id_fno_group, id_view_object, id_morphotype, np.name as name_project
            FROM {SCHEMA}.project_card pc
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            WHERE pc.id=%s
        """, [self.current_project_id])
        if not row:
            return
        prefix = self._file_prefix(row['id_fno_group'], row['id_view_object'], row['id_morphotype'], row['name_project'])

        paths = []
        for i in range(self.listImages.count()):
            p = self.listImages.item(i).data(QtCore.Qt.UserRole)
            if p: paths.append(p)
        if not paths:
            return
        if len(paths) > 3:
            paths = paths[:3]
        saved = 0
        for idx, src in enumerate(paths, start=1):
            dst = self._build_image_dest(prefix, idx, src)
            ensure_dir(Path(dst).parent)
            try:
                if str(Path(src).resolve()) != str(Path(dst).resolve()):
                    shutil.copy2(src, dst)
            except Exception as e:
                self._info(f'Не удалось скопировать {src}: {e}')
                continue
            self.db.execute(f"""
                INSERT INTO {SCHEMA}.project_image(project_id, path, priority)
                VALUES (%s,%s,%s)
                ON CONFLICT (project_id, priority) DO UPDATE SET path=EXCLUDED.path
            """, [self.current_project_id, str(dst), idx], commit=True)
            saved += 1
        if saved:
            self._info(f'Изображений сохранено: {saved}')

    def _save_tep_from_agr_if_any(self, project_id, id_name_project, x, y):
        r = self.db.fetchone(f"""
            SELECT id, area, spp_all, spp_live, spp_unlive
            FROM {SCHEMA}.agr
            WHERE ST_Intersects(geom, ST_SetSRID(ST_Point(%s,%s),4326))
            ORDER BY id
            LIMIT 1
        """, [x,y])
        if not r: return
        row = self.db.fetchone(f"""
            SELECT id FROM {SCHEMA}.tep_project WHERE id_name_project=%s AND id_project=%s
        """, [id_name_project, project_id])
        if row:
            self.db.execute(f"""
                UPDATE {SCHEMA}.tep_project
                SET area=%s, spp_all=%s, spp_live=%s, spp_unlive=%s
                WHERE id=%s
            """, [r['area'], r['spp_all'], r['spp_live'], r['spp_unlive'], row['id']], commit=True)
        else:
            self.db.execute(f"""
                INSERT INTO {SCHEMA}.tep_project(id_name_project, id_project, area, spp_all, spp_live, spp_unlive)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, [id_name_project, project_id, r['area'], r['spp_all'], r['spp_live'], r['spp_unlive']], commit=True)
        self.db.execute(f"""
            UPDATE {SCHEMA}.project_card
            SET area=COALESCE(area,%s),
                spp=COALESCE(spp,%s)
            WHERE id=%s
        """, [r['area'], r['spp_all'], project_id], commit=True)

    # ---------- Экспорт PDF ----------
    def _export_pdf(self):
        if not self.current_project_id:
            QtWidgets.QMessageBox.warning(self, 'PDF', 'Сначала сохраните проект.')
            return
        r = self.db.fetchone(f"""
            SELECT pc.*,
                   np.name as name_project,
                   fno.name as fno_group,
                   vo.name as view_object,
                   mt.name as morphotype,
                   ty.name as typology,
                   vr.name as vri
            FROM {SCHEMA}.project_card pc
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            LEFT JOIN {SCHEMA}.fno_group fno ON fno.id=pc.id_fno_group
            LEFT JOIN {SCHEMA}.view_object vo ON vo.id=pc.id_view_object
            LEFT JOIN {SCHEMA}.morphotype mt ON mt.id=pc.id_morphotype
            LEFT JOIN {SCHEMA}.typology ty ON ty.id=pc.id_typology
            LEFT JOIN {SCHEMA}.vri vr ON vr.id=pc.vri_id
            WHERE pc.id=%s
        """, [self.current_project_id])
        if not r:
            return
        tags = self.db.fetchall(f"""
            SELECT t.name FROM {SCHEMA}.project_teg pt
            JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
            WHERE pt.project_id=%s
            ORDER BY t.name
        """, [self.current_project_id])
        tags_text = ', '.join([x['name'] for x in tags])
        imgs = {1:'',2:'',3:''}
        rows = self.db.fetchall(f"""
            SELECT path, priority FROM {SCHEMA}.project_image
            WHERE project_id=%s
        """, [self.current_project_id])
        for im in rows:
            pr = int(im['priority'])
            if pr in (1,2,3) and not imgs[pr]:
                imgs[pr] = im['path']

        project = QgsProject.instance()
        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        if not Path(QPT_PATH).exists():
            QtWidgets.QMessageBox.warning(self, 'QPT', 'Не найден шаблон QPT.')
            return
        with open(QPT_PATH, 'r', encoding='utf-8') as f:
            tmpl = f.read()
        doc = QtXml.QDomDocument()
        ok = doc.setContent(tmpl)
        if not ok:
            QtWidgets.QMessageBox.warning(self, 'QPT', 'Ошибка разбора QPT.')
            return
        ctx = QgsReadWriteContext()
        layout.loadFromTemplate(doc, ctx)

        def set_label(item_id, text):
            it = layout.itemById(item_id)
            if isinstance(it, QgsLayoutItemLabel):
                it.setText('' if text is None else str(text))

        def set_image(item_id, path):
            it = layout.itemById(item_id)
            if isinstance(it, QgsLayoutItemPicture) and path:
                it.setPicturePath(path); it.refresh()

        set_label('name', r['name_project'])
        set_image('1', imgs.get(1, ''))
        set_image('2', imgs.get(2, ''))
        set_image('3', imgs.get(3, ''))
        set_label('fno', r['fno_group'])
        set_label('morphotype', r['morphotype'])
        set_label('view_object', r['view_object'])
        set_label('vri', r['vri'])
        set_label('typology', r['typology'])
        set_label('teg', tags_text)
        set_label('area', self._fmt_num(r['area']))
        set_label('footprint', self._fmt_num(r['footprint']))
        set_label('status', r['status'])
        set_label('spp', self._fmt_num(r['spp']))
        set_label('year_entry', '' if r['year_entry'] is None else str(r['year_entry']))
        set_label('mean_level', self._fmt_num(r['mean_level']))
        set_label('mean_height', self._fmt_num(r['mean_height']))
        set_label('density', self._fmt_num(r['density']))
        set_label('description', r['description'])

        prefix = self._file_prefix(r['id_fno_group'], r['id_view_object'], r['id_morphotype'], r['name_project'])
        pdf_path = str(Path(PDF_OUT_DIR) / f"{prefix}.pdf")
        exporter = QgsLayoutExporter(layout)
        res = exporter.exportToPdf(pdf_path, QgsLayoutExporter.PdfExportSettings())
        if res == QgsLayoutExporter.Success:
            self._info(f'PDF: {pdf_path}')
            if self.chkOpenPDF.isChecked():
                open_in_file_manager(pdf_path)
        else:
            QtWidgets.QMessageBox.warning(self, 'PDF', f'Ошибка сохранения PDF ({res}).')

    # ---------- Датасет ----------
    def _build_dataset_append(self):
        rows = self.db.fetchall(f"""
            SELECT pi.project_id, pi.path, pi.priority,
                   pc.id_fno_group, pc.id_view_object, pc.id_morphotype,
                   np.name as name_project,
                   pc.area, pc.footprint, pc.spp, pc.year_entry, pc.mean_level, pc.mean_height, pc.density,
                   fno.name as fno_group, vo.name as view_object, mt.name as morphotype, ty.name as typology, vr.name as vri
            FROM {SCHEMA}.project_image pi
            JOIN {SCHEMA}.project_card pc ON pc.id=pi.project_id
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            LEFT JOIN {SCHEMA}.fno_group fno ON fno.id=pc.id_fno_group
            LEFT JOIN {SCHEMA}.view_object vo ON vo.id=pc.id_view_object
            LEFT JOIN {SCHEMA}.morphotype mt ON mt.id=pc.id_morphotype
            LEFT JOIN {SCHEMA}.typology ty ON ty.id=pc.id_typology
            LEFT JOIN {SCHEMA}.vri vr ON vr.id=pc.vri_id
        """)
        tagmap = self.db.fetchall(f"""
            SELECT pt.project_id, array_agg(t.name ORDER BY t.name) as tags
            FROM {SCHEMA}.project_teg pt
            JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
            GROUP BY pt.project_id
        """)
        tagdict = {r['project_id']: ','.join(r['tags'] or []) for r in tagmap}
        hdr = ['project_id','dst_path','priority','name_project','tags',
               'fno_group','view_object','morphotype','typology','vri',
               'area','footprint','spp','year_entry','mean_level','mean_height','density']
        write_header = not Path(DATASET_META_CSV).exists()
        progress = QtWidgets.QProgressDialog('Обновление датасета...', 'Отмена', 0, len(rows), self)
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.show()
        with open(DATASET_META_CSV, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(hdr)
            new_rows = 0
            for i, r in enumerate(rows):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()
                prefix = self._file_prefix(r['id_fno_group'], r['id_view_object'], r['id_morphotype'], r['name_project'])
                dst = self._build_image_dest(prefix, int(r['priority'] or 3), r['path'])
                try:
                    if not Path(dst).exists():
                        ensure_dir(Path(dst).parent)
                        shutil.copy2(r['path'], dst)
                except Exception:
                    pass
                key = (r['project_id'], dst, int(r['priority'] or 3))
                if key in self.session_dataset_rows:
                    continue
                self.session_dataset_rows.add(key)
                w.writerow([
                    r['project_id'], dst, r['priority'] or '',
                    r['name_project'] or '', tagdict.get(r['project_id'], ''),
                    r['fno_group'] or '', r['view_object'] or '', r['morphotype'] or '', r['typology'] or '', r['vri'] or '',
                    r['area'] or '', r['footprint'] or '', r['spp'] or '', r['year_entry'] or '',
                    r['mean_level'] or '', r['mean_height'] or '', r['density'] or ''
                ])
                new_rows += 1
        progress.close()
        self._info(f'Датасет обновлен (append). Новых строк: {new_rows}')

    def _build_dataset_full(self):
        rows = self.db.fetchall(f"""
            SELECT pi.project_id, pi.path, pi.priority,
                   pc.id_fno_group, pc.id_view_object, pc.id_morphotype,
                   np.name as name_project,
                   pc.area, pc.footprint, pc.spp, pc.year_entry, pc.mean_level, pc.mean_height, pc.density,
                   fno.name as fno_group, vo.name as view_object, mt.name as morphotype, ty.name as typology, vr.name as vri
            FROM {SCHEMA}.project_image pi
            JOIN {SCHEMA}.project_card pc ON pc.id=pi.project_id
            LEFT JOIN {SCHEMA}.name_project np ON np.id=pc.id_name_project
            LEFT JOIN {SCHEMA}.fno_group fno ON fno.id=pc.id_fno_group
            LEFT JOIN {SCHEMA}.view_object vo ON vo.id=pc.id_view_object
            LEFT JOIN {SCHEMA}.morphotype mt ON mt.id=pc.id_morphotype
            LEFT JOIN {SCHEMA}.typology ty ON ty.id=pc.id_typology
            LEFT JOIN {SCHEMA}.vri vr ON vr.id=pc.vri_id
        """)
        tagmap = self.db.fetchall(f"""
            SELECT pt.project_id, array_agg(t.name ORDER BY t.name) as tags
            FROM {SCHEMA}.project_teg pt
            JOIN {SCHEMA}.teg t ON t.id=pt.teg_id
            GROUP BY pt.project_id
        """)
        tagdict = {r['project_id']: ','.join(r['tags'] or []) for r in tagmap}
        hdr = ['project_id','dst_path','priority','name_project','tags',
               'fno_group','view_object','morphotype','typology','vri',
               'area','footprint','spp','year_entry','mean_level','mean_height','density']
        progress = QtWidgets.QProgressDialog('Регенерация датасета...', 'Отмена', 0, len(rows), self)
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.show()
        with open(DATASET_META_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(hdr)
            cnt = 0
            for i, r in enumerate(rows):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()
                prefix = self._file_prefix(r['id_fno_group'], r['id_view_object'], r['id_morphotype'], r['name_project'])
                dst = self._build_image_dest(prefix, int(r['priority'] or 3), r['path'])
                try:
                    if not Path(dst).exists():
                        ensure_dir(Path(dst).parent)
                        shutil.copy2(r['path'], dst)
                except Exception:
                    pass
                w.writerow([
                    r['project_id'], dst, r['priority'] or '',
                    r['name_project'] or '', tagdict.get(r['project_id'], ''),
                    r['fno_group'] or '', r['view_object'] or '', r['morphotype'] or '', r['typology'] or '', r['vri'] or '',
                    r['area'] or '', r['footprint'] or '', r['spp'] or '', r['year_entry'] or '',
                    r['mean_level'] or '', r['mean_height'] or '', r['density'] or ''
                ])
                cnt += 1
        progress.close()
        self.session_dataset_rows.clear()
        self._info(f'CSV датасета регенерирован. Строк: {cnt}')

    # ---------- Слой проектов в QGIS ----------
    def _load_projects_layer(self):
        uri = (
            f"dbname='{PG['dbname']}' host='{PG['host']}' port='{PG['port']}' "
            f"user='{PG['user']}' password='{PG['password']}' sslmode=disable key='id' "
            f"srid=4326 type=Point table=\"{SCHEMA}\".\"project_card\" (geom)"
        )
        layer_name = 'UrbanView Projects'
        if self.projectsLayer and self.projectsLayer.isValid():
            QgsProject.instance().removeMapLayer(self.projectsLayer.id())
            self.projectsLayer = None
        vlayer = QgsVectorLayer(uri, layer_name, 'postgres')
        if not vlayer.isValid():
            QtWidgets.QMessageBox.warning(self, 'Слой', 'Не удалось загрузить слой проектов.')
            return
        QgsProject.instance().addMapLayer(vlayer)
        self.projectsLayer = vlayer
        self._info('Слой проектов загружен в QGIS.')

    # ---------- Экспорт/импорт/сброс Q-таблицы ----------
    def _export_q_table_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить Q-таблицу', '', 'JSON (*.json)')
        if not path:
            return
        rows = self.db.fetchall(f"SELECT state_hash, q_values, visit_count, updated_at FROM {SCHEMA}.rl_q_table")
        data = [{'state_hash': r['state_hash'], 'q_values': r['q_values'], 'visit_count': r['visit_count'], 'updated_at': str(r.get('updated_at') or '')} for r in rows]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._info(f'Q-таблица экспортирована: {path}')

    def _import_q_table_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Выбрать JSON Q-таблицы', '', 'JSON (*.json)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Импорт', f'Ошибка чтения JSON: {e}')
            return
        count = 0
        for rec in data:
            sh = rec.get('state_hash')
            qv = rec.get('q_values', {})
            vc = rec.get('visit_count', 0)
            if not sh:
                continue
            rc = self.db.execute_rowcount(f"UPDATE {SCHEMA}.rl_q_table SET q_values=%s, visit_count=%s, updated_at=NOW() WHERE state_hash=%s",
                                          [json.dumps(self.rl._normalize_qv(qv), ensure_ascii=False), int(vc or 0), sh], commit=True)
            if rc == 0:
                self.db.execute(f"INSERT INTO {SCHEMA}.rl_q_table(state_hash, q_values, visit_count, updated_at) VALUES (%s,%s,%s,NOW())",
                                [sh, json.dumps(self.rl._normalize_qv(qv), ensure_ascii=False), int(vc or 0)], commit=True)
            count += 1
        self._fill_q_table_preview()
        self._update_train_status()
        self._info(f'Q-таблица импортирована. Записей: {count}')

    def _reset_q_table(self):
        if QtWidgets.QMessageBox.question(self, 'Сброс', 'Удалить ВСЕ записи из Q-таблицы? Это необратимо.', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        self.db.execute(f"TRUNCATE TABLE {SCHEMA}.rl_q_table RESTART IDENTITY", [], commit=True)
        self.rl._meta_set('last_train_ts', '')
        self.rl._meta_set('last_train_updates', 0)
        self._fill_q_table_preview()
        self._update_train_status()
        self._info('Q-таблица очищена.')

    # ---------- Вспомогательное ----------
    def _update_summary(self):
        name = self.edName.text() or ''
        xy = f"x={self.edX.text()} y={self.edY.text()}"
        status = self.cbStatus.currentText() or ''
        year = '' if self.spYear.value()==self.spYear.minimum() and self.spYear.specialValueText() else str(self.spYear.value())
        tags = ', '.join(self._collect_tags_selected())
        imgs = [self.listImages.item(i).data(QtCore.Qt.UserRole) for i in range(self.listImages.count())]
        dirty = ' [изменено]' if self.dirty else ''
        self.txtSummary.setText(
            f"Проект: {name}{dirty}\n"
            f"Коорд.: {xy}\n"
            f"Статус/Год: {status} / {year}\n"
            f"Тэги: {tags}\n"
            f"Изображения: {len(imgs)}"
        )
        self._fill_q_table_preview()

    def _on_coord_changed(self, *args):
        self._mark_dirty()
        self._update_preview_marker()

    def _update_preview_marker(self):
        if not self.previewCanvas:
            return
        x = self._get_float(self.edX.text())
        y = self._get_float(self.edY.text())
        if x is None or y is None:
            if self.previewMarker:
                self.previewMarker.hide()
            return
        try:
            if not self.previewMarker:
                self.previewMarker = QgsVertexMarker(self.previewCanvas)
                self.previewMarker.setIconType(QgsVertexMarker.ICON_CROSS)
                self.previewMarker.setColor(QtGui.QColor(200, 0, 0))
                self.previewMarker.setIconSize(14)
                self.previewMarker.setPenWidth(3)
            self.previewMarker.setCenter(QgsPointXY(x, y))
            dx = 0.003
            rect = QgsRectangle(x - dx, y - dx, x + dx, y + dx)
            self.previewCanvas.setExtent(rect)
            self.previewCanvas.refresh()
        except Exception:
            pass

    def _file_prefix(self, id_fno, id_vo, id_mt, name_project):
        def clean(s):
            s = (s or '').strip().replace(' ', '_')
            for b in '<>:"/\|?*':
                s = s.replace(b, '')
            return s
        return f"{id_fno or 0}_{id_vo or 0}_{id_mt or 0}_{clean(name_project)}"

    def _build_image_dest(self, prefix, priority, src_path):
        ext = Path(src_path).suffix.lower() or '.jpg'
        return str(Path(DATASET_IMG_DIR) / f"{prefix}_{priority}{ext}")

    def _set_combo_by_id(self, cb, vid):
        idx = cb.findData(vid)
        cb.setCurrentIndex(idx if idx >= 0 else 0)

    def _set_spin(self, sp, v):
        if v is None:
            sp.setValue(sp.minimum())
        else:
            sp.setValue(float(v))

    def _get_float(self, s):
        s = (s or '').strip()
        if not s:
            return None
        try:
            return float(s.replace(',', '.'))
        except:
            return None

    def _nz(self, v, z=0.0):
        return float(v) if v is not None else z

    def _val_or_null(self, sp, intv=False):
        if isinstance(sp, QtWidgets.QSpinBox):
            v = sp.value()
            if v == sp.minimum() and sp.specialValueText():
                return None
            return int(v)
        v = sp.value()
        if sp.specialValueText() and abs(v - sp.minimum()) < 1e-9:
            return None
        return float(v)

    def _spin_val(self, sp, intv=False):
        if isinstance(sp, QtWidgets.QSpinBox) and intv:
            v = sp.value()
            if v == sp.minimum() and sp.specialValueText():
                return None
            return v
        v = sp.value()
        if sp.specialValueText() and abs(v - sp.minimum()) < 1e-9:
            return None
        return v

    def _fmt_num(self, v):
        if v is None:
            return ''
        try:
            return f'{float(v):.2f}'
        except:
            return str(v)

    def _info(self, msg):
        self.status.setText(msg)
        QtCore.QTimer.singleShot(5000, lambda: self.status.setText(''))
ui = UrbanViewWidget(); ui.show()
