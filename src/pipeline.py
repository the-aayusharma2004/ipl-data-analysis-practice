"""End-to-end IPL data pipeline.

Runs every stage in order, reusing each module's building blocks:

    raw CSV --ingestion--> frames
            --validation--> report (non-blocking)
            --transform--> cleaned "silver" frames  --load--> data/processed/*.parquet
            --star_schema--> "gold" dims + fact       --load--> data/processed/gold/*.parquet
            --visualize--> analysis + charts          -------> reports/*.png

The Spark-dependent stages share one session; analysis/visualization run on
pandas over the gold Parquet and need no Spark.
"""

from __future__ import annotations

import logging

from spark_session import create_spark_session

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the full pipeline end to end."""
    import star_schema
    import transform
    import visualize
    from ingestion import configure_logging, load_all
    from load import GOLD_DIR, PROCESSED_DIR, write_all
    from validation import validate_all

    configure_logging()
    spark = create_spark_session()
    try:
        logger.info("STAGE 1/5: ingest raw CSVs")
        frames = load_all(spark)
        if not frames:
            logger.error("No datasets loaded; aborting pipeline")
            return

        logger.info("STAGE 2/5: validate (report only)")
        validate_all(frames)

        logger.info("STAGE 3/5: transform -> silver Parquet")
        cleaned = transform.transform_all(frames)
        write_all(cleaned, PROCESSED_DIR)

        logger.info("STAGE 4/5: build star schema -> gold Parquet")
        star = star_schema.build_star(cleaned)
        write_all(star, GOLD_DIR)
    finally:
        spark.stop()

    # Analysis + charts run on pandas over the gold Parquet (no Spark needed).
    logger.info("STAGE 5/5: analysis + visualizations -> reports")
    visualize.main()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
