#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import posixpath
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET


PROJECT = Path(
    "."
)

INPUT_FILE = (
    PROJECT
    / "01_raw_data/perturbational_validation/EPA_HTTr/"
      "HTTr signature catalog for dashboard "
      "2022-06-24_firstSheetOnly.xlsx"
)

TABLE_DIR = (
    PROJECT
    / "07_tables/main_tables/phase6"
)

METADATA_DIR = (
    PROJECT
    / "02_metadata/phase6/EPA_HTTr_schema"
)

LOG_FILE = (
    PROJECT
    / "09_pipeline_logs/phase6/"
      "phase6B4A_XLSX_schema_inspection.log"
)

SHEET_MANIFEST_FILE = (
    TABLE_DIR
    / "phase6B4A_XLSX_sheet_manifest.tsv"
)

PREVIEW_FILE = (
    TABLE_DIR
    / "phase6B4A_XLSX_first_rows_long.tsv"
)

COMPLETION_FILE = (
    TABLE_DIR
    / "phase6B4A_completion_summary.tsv"
)

REPORT_FILE = (
    METADATA_DIR
    / "phase6B4A_XLSX_preview_report.txt"
)

MAIN_NS = (
    "http://schemas.openxmlformats.org/"
    "spreadsheetml/2006/main"
)

DOCUMENT_REL_NS = (
    "http://schemas.openxmlformats.org/"
    "officeDocument/2006/relationships"
)

PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/"
    "package/2006/relationships"
)

for directory in [
    TABLE_DIR,
    METADATA_DIR,
    LOG_FILE.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in columns
                }
            )


def all_text(
    element: ET.Element | None,
) -> str:
    if element is None:
        return ""

    return "".join(
        text_node.text or ""
        for text_node in element.iter()
        if text_node.tag.endswith(
            "}t"
        )
    )


def load_shared_strings(
    archive: zipfile.ZipFile,
) -> list[str]:
    path = "xl/sharedStrings.xml"

    if path not in archive.namelist():
        return []

    root = ET.fromstring(
        archive.read(
            path
        )
    )

    strings = []

    for item in root.findall(
        f"{{{MAIN_NS}}}si"
    ):
        strings.append(
            all_text(
                item
            )
        )

    return strings


def decode_cell(
    cell: ET.Element,
    shared_strings: list[str],
) -> tuple[str, str, str]:
    cell_type = cell.attrib.get(
        "t",
        ""
    )

    formula_element = cell.find(
        f"{{{MAIN_NS}}}f"
    )

    value_element = cell.find(
        f"{{{MAIN_NS}}}v"
    )

    inline_element = cell.find(
        f"{{{MAIN_NS}}}is"
    )

    formula = (
        formula_element.text or ""
        if formula_element is not None
        else ""
    )

    raw_value = (
        value_element.text or ""
        if value_element is not None
        else ""
    )

    if cell_type == "s":
        try:
            value = shared_strings[
                int(
                    raw_value
                )
            ]

        except (
            ValueError,
            IndexError,
        ):
            value = raw_value

    elif cell_type == "inlineStr":
        value = all_text(
            inline_element
        )

    elif cell_type == "b":
        value = (
            "TRUE"
            if raw_value == "1"
            else "FALSE"
        )

    else:
        value = raw_value

    value = re.sub(
        r"[\t\r\n]+",
        " ",
        value,
    )

    formula = re.sub(
        r"[\t\r\n]+",
        " ",
        formula,
    )

    return (
        cell_type,
        value,
        formula,
    )


def normalized_sheet_target(
    target: str,
) -> str:
    if target.startswith(
        "/"
    ):
        return target.lstrip(
            "/"
        )

    return posixpath.normpath(
        posixpath.join(
            "xl",
            target,
        )
    )


def load_rdata_object_count() -> int:
    manifest = (
        TABLE_DIR
        / "phase6B4A_RData_object_manifest.tsv"
    )

    if not manifest.exists():
        return 0

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        return sum(
            1
            for _ in reader
        )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing XLSX source: {INPUT_FILE}"
        )

    report_lines = [
        "PHASE 6B4A EPA HTTr XLSX SCHEMA REPORT",
        "",
        f"Input: {INPUT_FILE}",
        f"File size: {INPUT_FILE.stat().st_size} bytes",
        "",
    ]

    sheet_rows: list[
        dict[str, object]
    ] = []

    preview_rows: list[
        dict[str, object]
    ] = []

    with zipfile.ZipFile(
        INPUT_FILE,
        mode="r",
    ) as archive:
        archive_names = set(
            archive.namelist()
        )

        workbook_root = ET.fromstring(
            archive.read(
                "xl/workbook.xml"
            )
        )

        relationships_root = ET.fromstring(
            archive.read(
                "xl/_rels/workbook.xml.rels"
            )
        )

        relationship_targets = {
            relationship.attrib[
                "Id"
            ]: relationship.attrib[
                "Target"
            ]
            for relationship
            in relationships_root.findall(
                f"{{{PACKAGE_REL_NS}}}Relationship"
            )
        }

        shared_strings = load_shared_strings(
            archive
        )

        sheets = workbook_root.find(
            f"{{{MAIN_NS}}}sheets"
        )

        if sheets is None:
            raise RuntimeError(
                "No worksheet records found."
            )

        for sheet_index, sheet in enumerate(
            list(
                sheets
            ),
            start=1,
        ):
            sheet_name = sheet.attrib.get(
                "name",
                f"Sheet{sheet_index}",
            )

            relationship_id = sheet.attrib.get(
                f"{{{DOCUMENT_REL_NS}}}id",
                "",
            )

            target = relationship_targets.get(
                relationship_id,
                "",
            )

            worksheet_path = normalized_sheet_target(
                target
            )

            if worksheet_path not in archive_names:
                raise FileNotFoundError(
                    (
                        "Worksheet XML is missing: "
                        f"{worksheet_path}"
                    )
                )

            root = ET.fromstring(
                archive.read(
                    worksheet_path
                )
            )

            dimension_element = root.find(
                f"{{{MAIN_NS}}}dimension"
            )

            dimension_reference = (
                dimension_element.attrib.get(
                    "ref",
                    "",
                )
                if dimension_element is not None
                else ""
            )

            sheet_data = root.find(
                f"{{{MAIN_NS}}}sheetData"
            )

            row_count = 0
            cell_count = 0
            formula_count = 0
            nonempty_row_count = 0
            previewed_nonempty_rows = 0

            report_lines.extend(
                [
                    (
                        "============================================================"
                    ),
                    f"SHEET {sheet_index}: {sheet_name}",
                    f"Worksheet path: {worksheet_path}",
                    f"Dimension reference: {dimension_reference}",
                    "",
                ]
            )

            if sheet_data is not None:
                for row in list(
                    sheet_data
                ):
                    row_count += 1

                    row_number = int(
                        row.attrib.get(
                            "r",
                            row_count,
                        )
                    )

                    decoded_cells = []

                    for cell in list(
                        row
                    ):
                        if not cell.tag.endswith(
                            "}c"
                        ):
                            continue

                        cell_count += 1

                        reference = cell.attrib.get(
                            "r",
                            "",
                        )

                        (
                            cell_type,
                            value,
                            formula,
                        ) = decode_cell(
                            cell,
                            shared_strings,
                        )

                        if formula:
                            formula_count += 1

                        if value or formula:
                            decoded_cells.append(
                                {
                                    "sheet_index":
                                        sheet_index,

                                    "sheet_name":
                                        sheet_name,

                                    "row_number":
                                        row_number,

                                    "cell_reference":
                                        reference,

                                    "cell_type":
                                        cell_type,

                                    "formula":
                                        formula,

                                    "value":
                                        value,
                                }
                            )

                    if decoded_cells:
                        nonempty_row_count += 1

                        if previewed_nonempty_rows < 15:
                            preview_rows.extend(
                                decoded_cells
                            )

                            previewed_nonempty_rows += 1

                            row_text = " | ".join(
                                (
                                    f"{item['cell_reference']}="
                                    f"{item['value']}"
                                    + (
                                        f" [formula={item['formula']}]"
                                        if item[
                                            "formula"
                                        ]
                                        else ""
                                    )
                                )
                                for item in decoded_cells
                            )

                            report_lines.append(
                                row_text
                            )

            merge_cells = root.find(
                f"{{{MAIN_NS}}}mergeCells"
            )

            merged_cell_ranges = (
                len(
                    list(
                        merge_cells
                    )
                )
                if merge_cells is not None
                else 0
            )

            sheet_rows.append(
                {
                    "sheet_index":
                        sheet_index,

                    "sheet_name":
                        sheet_name,

                    "relationship_id":
                        relationship_id,

                    "worksheet_path":
                        worksheet_path,

                    "dimension_reference":
                        dimension_reference,

                    "XML_row_records":
                        row_count,

                    "nonempty_rows":
                        nonempty_row_count,

                    "cell_records":
                        cell_count,

                    "formula_cells":
                        formula_count,

                    "merged_cell_ranges":
                        merged_cell_ranges,

                    "previewed_nonempty_rows":
                        previewed_nonempty_rows,
                }
            )

            report_lines.append(
                ""
            )

    sheet_columns = [
        "sheet_index",
        "sheet_name",
        "relationship_id",
        "worksheet_path",
        "dimension_reference",
        "XML_row_records",
        "nonempty_rows",
        "cell_records",
        "formula_cells",
        "merged_cell_ranges",
        "previewed_nonempty_rows",
    ]

    preview_columns = [
        "sheet_index",
        "sheet_name",
        "row_number",
        "cell_reference",
        "cell_type",
        "formula",
        "value",
    ]

    write_tsv(
        SHEET_MANIFEST_FILE,
        sheet_rows,
        sheet_columns,
    )

    write_tsv(
        PREVIEW_FILE,
        preview_rows,
        preview_columns,
    )

    REPORT_FILE.write_text(
        "\n".join(
            report_lines
        )
        + "\n",
        encoding="utf-8",
    )

    rdata_object_count = (
        load_rdata_object_count()
    )

    completion_rows = [
        {
            "RData_objects_loaded":
                rdata_object_count,

            "XLSX_sheets":
                len(
                    sheet_rows
                ),

            "XLSX_total_nonempty_rows":
                sum(
                    int(
                        row[
                            "nonempty_rows"
                        ]
                    )
                    for row in sheet_rows
                ),

            "XLSX_total_cells":
                sum(
                    int(
                        row[
                            "cell_records"
                        ]
                    )
                    for row in sheet_rows
                ),

            "XLSX_formula_cells":
                sum(
                    int(
                        row[
                            "formula_cells"
                        ]
                    )
                    for row in sheet_rows
                ),

            "source_files_modified":
                False,

            "Phase6B4A_status":
                (
                    "completed"
                    if (
                        rdata_object_count > 0
                        and len(
                            sheet_rows
                        ) > 0
                    )
                    else "failed_schema_inspection"
                ),
        }
    ]

    write_tsv(
        COMPLETION_FILE,
        completion_rows,
        [
            "RData_objects_loaded",
            "XLSX_sheets",
            "XLSX_total_nonempty_rows",
            "XLSX_total_cells",
            "XLSX_formula_cells",
            "source_files_modified",
            "Phase6B4A_status",
        ],
    )

    LOG_FILE.write_text(
        "\n".join(
            report_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "===== XLSX SHEET MANIFEST ====="
    )

    for row in sheet_rows:
        print(
            json.dumps(
                row,
                ensure_ascii=False,
            )
        )

    print()
    print(
        "===== PHASE 6B4A COMPLETION ====="
    )

    print(
        json.dumps(
            completion_rows[0],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            (
                "Phase 6B4A XLSX inspection failed: "
                f"{type(error).__name__}: {error}"
            ),
            file=sys.stderr,
        )

        raise
