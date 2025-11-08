BEGIN;

-- Схема и расширение PostGIS
CREATE SCHEMA IF NOT EXISTS urbanview;

CREATE EXTENSION IF NOT EXISTS postgis;

-- СПРАВОЧНИКИ
CREATE TABLE IF NOT EXISTS urbanview.fno_group (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.view_object (
  id            SERIAL PRIMARY KEY,
  fno_group_id  INTEGER REFERENCES urbanview.fno_group(id)
                 ON UPDATE CASCADE ON DELETE SET NULL,
  name          TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.morphotype (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.typology (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.name_project (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.priority (
  id    SERIAL PRIMARY KEY,
  value INTEGER NOT NULL CHECK (value BETWEEN 1 AND 3),
  UNIQUE (value)
);

CREATE TABLE IF NOT EXISTS urbanview.vri (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS urbanview.teg (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

-- AGR (полигональный слой)
CREATE TABLE IF NOT EXISTS urbanview.agr (
  id         SERIAL PRIMARY KEY,
  name       TEXT,
  area       DOUBLE PRECISION,
  year       INTEGER CHECK (year BETWEEN 1800 AND 2100),
  spp_all    DOUBLE PRECISION,
  spp_live   DOUBLE PRECISION,
  spp_unlive DOUBLE PRECISION,
  geom       geometry(Polygon, 4326)
);

-- ОСНОВНАЯ КАРТОЧКА ПРОЕКТА
CREATE TABLE IF NOT EXISTS urbanview.project_card (
  id              SERIAL PRIMARY KEY,
  id_fno_group    INTEGER REFERENCES urbanview.fno_group(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  id_view_object  INTEGER REFERENCES urbanview.view_object(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  id_morphotype   INTEGER REFERENCES urbanview.morphotype(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  id_typology     INTEGER REFERENCES urbanview.typology(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  id_name_project INTEGER REFERENCES urbanview.name_project(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  priority_id     INTEGER REFERENCES urbanview.priority(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  id_agr          INTEGER REFERENCES urbanview.agr(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  vri_id          INTEGER REFERENCES urbanview.vri(id)
                   ON UPDATE CASCADE ON DELETE SET NULL,
  x            DOUBLE PRECISION,
  y            DOUBLE PRECISION,
  status       TEXT CHECK (status IN ('построен','проект','строительство','заброшен')),
  area         DOUBLE PRECISION,
  spp          DOUBLE PRECISION,  -- СПП из tep_project (если заполняется)
  footprint    DOUBLE PRECISION,  -- площадь пятна застройки
  year_entry   INTEGER CHECK (year_entry BETWEEN 1800 AND 2100),
    mean_level   DOUBLE PRECISION,  -- средняя этажность
  mean_height  DOUBLE PRECISION,  -- средняя высота
  density      DOUBLE PRECISION,  -- плотность
  description  TEXT,
  geom         geometry(Point, 4326),
  image_path   TEXT
);

-- ТЭП по проекту
CREATE TABLE IF NOT EXISTS urbanview.tep_project (
  id               SERIAL PRIMARY KEY,
  id_name_project  INTEGER NOT NULL REFERENCES urbanview.name_project(id)
                    ON UPDATE CASCADE ON DELETE CASCADE,
  id_project       INTEGER NOT NULL REFERENCES urbanview.project_card(id)
                    ON UPDATE CASCADE ON DELETE CASCADE,
  area             DOUBLE PRECISION,
  spp_all          DOUBLE PRECISION,
  spp_live         DOUBLE PRECISION,
  spp_unlive       DOUBLE PRECISION,
  mean_level       DOUBLE PRECISION,
  mean_height      DOUBLE PRECISION,
  density          DOUBLE PRECISION
);

-- ИЗОБРАЖЕНИЯ
CREATE TABLE IF NOT EXISTS urbanview.project_image (
  id          SERIAL PRIMARY KEY,
  project_id  INTEGER NOT NULL REFERENCES urbanview.project_card(id)
               ON UPDATE CASCADE ON DELETE CASCADE,
  path        TEXT NOT NULL,
  priority    INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 3),
  UNIQUE (project_id, priority)
);

-- Q-learning
CREATE TABLE IF NOT EXISTS urbanview.rl_q_table (
  id           SERIAL PRIMARY KEY,
  state_hash   TEXT NOT NULL UNIQUE,
  q_values     JSONB NOT NULL,
  visit_count  INTEGER DEFAULT 0,
  created_at   TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
  updated_at   TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Связка проект—тег
CREATE TABLE IF NOT EXISTS urbanview.project_teg (
  project_id  INTEGER NOT NULL REFERENCES urbanview.project_card(id)
               ON UPDATE CASCADE ON DELETE CASCADE,
  teg_id      INTEGER NOT NULL REFERENCES urbanview.teg(id)
               ON UPDATE CASCADE ON DELETE CASCADE,
  PRIMARY KEY (project_id, teg_id)
);

-- Индексы (пространственные и для FK)
CREATE INDEX IF NOT EXISTS idx_agr_geom                  ON urbanview.agr USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_project_card_geom         ON urbanview.project_card USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_view_object_fno_group_id  ON urbanview.view_object (fno_group_id);
CREATE INDEX IF NOT EXISTS idx_pc_id_fno_group           ON urbanview.project_card (id_fno_group);
CREATE INDEX IF NOT EXISTS idx_pc_id_view_object         ON urbanview.project_card (id_view_object);
CREATE INDEX IF NOT EXISTS idx_pc_id_morphotype          ON urbanview.project_card (id_morphotype);
CREATE INDEX IF NOT EXISTS idx_pc_id_typology            ON urbanview.project_card (id_typology);
CREATE INDEX IF NOT EXISTS idx_pc_id_name_project        ON urbanview.project_card (id_name_project);
CREATE INDEX IF NOT EXISTS idx_pc_priority_id            ON urbanview.project_card (priority_id);
CREATE INDEX IF NOT EXISTS idx_pc_id_agr                 ON urbanview.project_card (id_agr);
CREATE INDEX IF NOT EXISTS idx_pc_vri_id                 ON urbanview.project_card (vri_id);
CREATE INDEX IF NOT EXISTS idx_tep_id_name_project       ON urbanview.tep_project (id_name_project);
CREATE INDEX IF NOT EXISTS idx_tep_id_project            ON urbanview.tep_project (id_project);

-- Триггер на обновление updated_at
CREATE OR REPLACE FUNCTION urbanview.tg_set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_rl_q_table_set_updated_at ON urbanview.rl_q_table;
CREATE TRIGGER trg_rl_q_table_set_updated_at
BEFORE UPDATE ON urbanview.rl_q_table
FOR EACH ROW
EXECUTE FUNCTION urbanview.tg_set_updated_at();

-- Комментарии
COMMENT ON TABLE  urbanview.agr IS 'Архитектурно-Градостроительные решения (полигональный слой)';
COMMENT ON COLUMN urbanview.project_card.status      IS 'Статус: построен | проект | строительство | заброшен';
COMMENT ON COLUMN urbanview.project_card.spp         IS 'СПП (может подтягиваться из urbanview.tep_project)';
COMMENT ON COLUMN urbanview.project_card.footprint   IS 'Площадь пятна застройки';

COMMIT;

 -- 1) rl_q_table: привести q_values к JSONB и хранить updated_at
ALTER TABLE urbanview.rl_q_table
    ADD COLUMN IF NOT EXISTS updated_at timestamp with time zone;

ALTER TABLE urbanview.rl_q_table
    ALTER COLUMN q_values TYPE jsonb
    USING CASE
        WHEN q_values IS NULL THEN '{}'::jsonb
        WHEN pg_typeof(q_values)::text IN ('json','jsonb') THEN q_values::jsonb
        ELSE to_jsonb(q_values)
    END;

ALTER TABLE urbanview.rl_q_table
    ALTER COLUMN q_values SET DEFAULT '{}'::jsonb;

UPDATE urbanview.rl_q_table SET updated_at = COALESCE(updated_at, NOW()) WHERE updated_at IS NULL;

-- 2) rl_meta: хранить метаданные RL
CREATE TABLE IF NOT EXISTS urbanview.rl_meta (
    key text PRIMARY KEY,
    value text
);


-- 3) Индексы (опционально)
CREATE INDEX IF NOT EXISTS idx_rl_q_table_updated_at ON urbanview.rl_q_table(updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_q_state_hash ON urbanview.rl_q_table(state_hash);

COMMIT;


 

