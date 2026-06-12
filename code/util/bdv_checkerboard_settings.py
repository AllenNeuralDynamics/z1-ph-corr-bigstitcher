from __future__ import annotations

import io
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import boto3


class CheckerboardSettingsBDV:
    def __init__(
        self,
        input_xml: str,
        output_xml: str,
        existing_settings: str | None = None,
        vis_grid: bool = True,
        s3_client=None,
    ):
        self.input_xml = input_xml
        self.output_xml = output_xml
        self.existing_settings = existing_settings
        self.vis_grid = vis_grid
        self.s3 = s3_client or boto3.client("s3")

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        return str(path).startswith("s3://")

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def _parse_xml(self, path: str) -> ET.ElementTree:
        if self._is_s3_path(path):
            bucket, key = self._parse_s3_uri(path)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            return ET.parse(obj["Body"])

        return ET.parse(path)

    def _write_xml(self, tree: ET.ElementTree, path: str) -> None:
        if self._is_s3_path(path):
            bucket, key = self._parse_s3_uri(path)

            buffer = io.BytesIO()
            tree.write(buffer, encoding="utf-8", xml_declaration=True)
            buffer.seek(0)

            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=buffer.getvalue(),
                ContentType="application/xml",
            )
            return

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        tree.write(path, encoding="utf-8", xml_declaration=True)

    def extract_grid_position_from_name(self, name: str):
        match = re.search(r"Tile_X_(\d+)_Y_(\d+)", name)
        if match:
            return int(match.group(1)), int(match.group(2))

        return None

    def generate_settings_file(self):
        GREEN = "-16711936"  # 0xFF00FF00 signed int
        RED = "-65536"       # 0xFFFF0000 signed int

        tree = self._parse_xml(self.input_xml)
        root = tree.getroot()

        view_setups = root.findall(".//ViewSetup")

        if not view_setups:
            print("ERROR: No ViewSetup elements found")
            return

        tile_data = []

        print(len(view_setups), "ViewSetups found:")

        for view_setup in view_setups:
            id_elem = view_setup.find("id")
            if id_elem is None or id_elem.text is None:
                print("  WARNING: ViewSetup missing id")
                continue

            raw_id = id_elem.text.strip()
            print(f"DEBUG: Raw id attribute = '{raw_id}'")

            setup_id = int(raw_id)
            name_elem = view_setup.find("name")

            if name_elem is None or name_elem.text is None:
                print(f"  WARNING: No name element for setup {setup_id}")
                continue

            name = name_elem.text
            grid_pos = self.extract_grid_position_from_name(name)

            if grid_pos:
                x_pos, y_pos = grid_pos
                tile_data.append((setup_id, x_pos, y_pos, name))
                print(f"  Setup {setup_id}: X={x_pos}, Y={y_pos} ({name})")
            else:
                print(f"  WARNING: Could not parse position from name: {name}")

        if not tile_data:
            print("ERROR: No tile positions found in names")
            return

        print(f"\nDetected {len(tile_data)} tiles")

        # Read existing settings if provided, preserving first converter min/max.
        min_val, max_val = "0.0", "30.0"

        if self.existing_settings:
            try:
                existing_tree = self._parse_xml(self.existing_settings)
                existing_root = existing_tree.getroot()
                first_setup = existing_root.find(".//ConverterSetup")

                if first_setup is not None:
                    min_elem = first_setup.find("min")
                    max_elem = first_setup.find("max")

                    if min_elem is not None and min_elem.text is not None:
                        min_val = min_elem.text

                    if max_elem is not None and max_elem.text is not None:
                        max_val = max_elem.text
            except Exception as e:
                print(f"WARNING: Could not read existing settings: {e}")

        settings_root = ET.Element("Settings")

        viewer_state = ET.SubElement(settings_root, "ViewerState")
        sources = ET.SubElement(viewer_state, "Sources")

        for _ in range(len(tile_data)):
            source = ET.SubElement(sources, "Source")
            active = ET.SubElement(source, "active")
            active.text = "true"

        source_groups = ET.SubElement(viewer_state, "SourceGroups")

        for i in range(min(10, len(tile_data))):
            group = ET.SubElement(source_groups, "SourceGroup")

            active = ET.SubElement(group, "active")
            active.text = "true"

            name = ET.SubElement(group, "name")
            name.text = f"group {i + 1}"

            group_id = ET.SubElement(group, "id")
            group_id.text = str(tile_data[i][0])

        display_mode = ET.SubElement(viewer_state, "DisplayMode")
        display_mode.text = "fs"

        interpolation = ET.SubElement(viewer_state, "Interpolation")
        interpolation.text = "nearestneighbor"

        current_source = ET.SubElement(viewer_state, "CurrentSource")
        current_source.text = "0"

        current_group = ET.SubElement(viewer_state, "CurrentGroup")
        current_group.text = "0"

        current_timepoint = ET.SubElement(viewer_state, "CurrentTimePoint")
        current_timepoint.text = "0"

        setup_assignments = ET.SubElement(settings_root, "SetupAssignments")
        converter_setups = ET.SubElement(setup_assignments, "ConverterSetups")

        print("\nApplying checkerboard pattern:")

        tile_data_sorted = sorted(tile_data, key=lambda x: x[0])

        for setup_id, x_pos, y_pos, name in tile_data_sorted:
            is_green = (x_pos + y_pos) % 2 == 0
            color_value = GREEN if is_green else RED

            setup = ET.SubElement(converter_setups, "ConverterSetup")

            id_elem = ET.SubElement(setup, "id")
            id_elem.text = str(setup_id)

            min_elem = ET.SubElement(setup, "min")
            min_elem.text = min_val

            max_elem = ET.SubElement(setup, "max")
            max_elem.text = max_val

            color_elem = ET.SubElement(setup, "color")
            color_elem.text = color_value

            group_id = ET.SubElement(setup, "groupId")
            group_id.text = "0"

            color_name = "GREEN" if is_green else "RED"
            print(f"Setup {setup_id:3d} at X={x_pos}, Y={y_pos} -> {color_name}")

        minmax_groups = ET.SubElement(setup_assignments, "MinMaxGroups")
        minmax_group = ET.SubElement(minmax_groups, "MinMaxGroup")

        id_elem = ET.SubElement(minmax_group, "id")
        id_elem.text = "0"

        full_range_min = ET.SubElement(minmax_group, "fullRangeMin")
        full_range_min.text = "-2.147483648E9"

        full_range_max = ET.SubElement(minmax_group, "fullRangeMax")
        full_range_max.text = "2.147483647E9"

        range_min = ET.SubElement(minmax_group, "rangeMin")
        range_min.text = "0.0"

        range_max = ET.SubElement(minmax_group, "rangeMax")
        range_max.text = "65535.0"

        current_min = ET.SubElement(minmax_group, "currentMin")
        current_min.text = "0.0"

        current_max = ET.SubElement(minmax_group, "currentMax")
        current_max.text = "65535.0"

        transforms = ET.SubElement(settings_root, "ManualSourceTransforms")

        for _ in range(len(tile_data)):
            transform = ET.SubElement(transforms, "SourceTransform")
            transform.set("type", "affine")

            affine = ET.SubElement(transform, "affine")
            affine.text = (
                "1.0 0.0 0.0 0.0 "
                "0.0 1.0 0.0 0.0 "
                "0.0 0.0 1.0 0.0"
            )

        ET.SubElement(settings_root, "Bookmarks")

        output_tree = ET.ElementTree(settings_root)
        self._write_xml(output_tree, self.output_xml)

        print("\n" + "=" * 60)
        print(f"Settings file created: {self.output_xml}")
        print(f"Total tiles: {len(tile_data)}")
        print("Colors applied based on X + Y grid position")

    def run(self):
        self.generate_settings_file()

        