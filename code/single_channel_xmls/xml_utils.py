import boto3
import re
import json
import xmltodict
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Union
from xml.dom import minidom
import copy
from pathlib import Path


class BigStitcherXMLManager:
    """
    Manager class for handling BigStitcher XML transformations between 
    single-channel stitched and multichannel camera-aligned datasets.
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def load_xml(self, xml_path: str) -> OrderedDict:
        """
        Load XML from local file or S3 path.
        
        Parameters
        ----------
        xml_path : str
            Path to XML file. Can be local path or S3 path (s3://bucket/path/file.xml).
            
        Returns
        -------
        OrderedDict
            Parsed XML content
        """
        if xml_path.startswith('s3://'):
            # Parse S3 path
            s3_path = xml_path.replace("s3://", "")
            bucket_name, key = s3_path.split("/", 1)
            
            # Read from S3
            try:
                response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
                xml_content = response['Body'].read().decode('utf-8')
            except Exception as e:
                raise Exception(f"Could not read XML file from S3: {e}")
        else:
            # Read from local file
            with open(xml_path, "r") as file:
                xml_content = file.read()
        
        # Parse XML content
        data = xmltodict.parse(xml_content)
        return data
    
    def save_xml(self, data: OrderedDict, output_path: str, pretty: bool = True) -> None:
        """
        Save XML data to file or S3.
        
        Parameters
        ----------
        data : OrderedDict
            XML data to save
        output_path : str
            Output path (local or S3)
        pretty : bool
            Whether to format XML with indentation
        """
        # Convert to XML string
        xml_str = xmltodict.unparse(data, pretty=pretty, indent='  ')
        
        if output_path.startswith('s3://'):
            # Save to S3
            s3_path = output_path.replace("s3://", "")
            bucket_name, key = s3_path.split("/", 1)
            
            try:
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=xml_str.encode('utf-8'),
                    ContentType='application/xml'
                )
                print(f"Saved XML to S3: {output_path}")
            except Exception as e:
                raise Exception(f"Could not save XML to S3: {e}")
        else:
            # Save to local file
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)

            with open(output_path, 'w') as f:
                f.write(xml_str)
            print(f"Saved XML to local file: {output_path}")
    
    def get_tile_id_from_name(self, data: dict, tilename: str) -> int:
        """
        Get tile ID from tile name in XML data.
        
        Parameters
        ----------
        data : dict
            Parsed XML data
        tilename : str
            Name of the tile to find
            
        Returns
        -------
        int
            Tile ID number
        """
        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
        
        # Ensure viewsetups is a list
        if not isinstance(viewsetups, list):
            viewsetups = [viewsetups]
        
        matching_viewsetup = [v for v in viewsetups if tilename in v.get('name', '')]
        
        if not matching_viewsetup:
            raise ValueError(f"No ViewSetup found for tile name: {tilename}")
        
        # Get tile_number from this viewsetup
        matching_tile_number = matching_viewsetup[0]['attributes']['tile']
        
        return int(matching_tile_number)
    
    def extract_tile_name_from_viewsetup(self, viewsetup: dict) -> str:
        """
        Extract tile name from ViewSetup entry.
        
        Parameters
        ----------
        viewsetup : dict
            ViewSetup dictionary from XML
            
        Returns
        -------
        str
            Extracted tile name
        """
        name = viewsetup.get('name', '')
        # Remove channel information if present
        # Common patterns: Tile_X_0001_Y_0002_Z_0003_ch_488
        pattern = r"(Tile_X_\d+_Y_\d+_Z_\d+)"
        match = re.search(pattern, name)
        if match:
            return match.group(1)
        return name
    
    def get_channel_from_viewsetup(self, viewsetup: dict) -> Optional[int]:
        """
        Extract channel wavelength from ViewSetup.
        
        Parameters
        ----------
        viewsetup : dict
            ViewSetup dictionary from XML
            
        Returns
        -------
        int or None
            Channel wavelength if found
        """
        name = viewsetup.get('name', '')
        pattern = r"(ch|CH)_(\d+)"
        match = re.search(pattern, name)
        if match:
            return int(match.group(2))
        return None
    
    def extract_transforms_for_tile(self, data: dict, tile_id: int) -> List[dict]:
        """
        Extract all transforms for a specific tile ID.
        
        Parameters
        ----------
        data : dict
            Parsed XML data
        tile_id : int
            Tile ID to extract transforms for
            
        Returns
        -------
        list
            List of transform dictionaries
        """
        view_registrations = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
        
        if not isinstance(view_registrations, list):
            view_registrations = [view_registrations]
        
        for view_reg in view_registrations:
            if int(view_reg.get("@setup", -1)) == tile_id:
                transforms = view_reg.get("ViewTransform", [])
                if not isinstance(transforms, list):
                    transforms = [transforms]
                return transforms
        
        return []
    
    def transfer_stitching_transforms(
        self,
        single_channel_xml_path: str,
        multichannel_xml_path: str,
        output_xml_path: str,
        transform_index: int = 0,
        channels_to_apply: Optional[List[int]] = None
    ) -> None:
        """
        Transfer stitching transforms from single-channel XML to multichannel XML.
        
        Parameters
        ----------
        single_channel_xml_path : str
            Path to single-channel stitched XML
        multichannel_xml_path : str
            Path to multichannel camera-aligned XML
        output_xml_path : str
            Path for output XML with combined transforms
        transform_index : int
            Which transform to extract from single-channel (default: 1 for second transform)
        channels_to_apply : list, optional
            List of channel wavelengths to apply transforms to. If None, applies to all.
        """
        print("Loading single-channel stitched XML...")
        single_channel_data = self.load_xml(single_channel_xml_path)
        
        print("Loading multichannel camera-aligned XML...")
        multichannel_data = self.load_xml(multichannel_xml_path)
        
        # Create a copy for modification
        output_data = copy.deepcopy(multichannel_data)
        
        # Build mapping of tile positions to transforms from single-channel
        print("Extracting transforms from single-channel XML...")
        single_channel_transforms = self._build_tile_transform_map(
            single_channel_data, 
            transform_index
        )
        
        print(f"Found transforms for {len(single_channel_transforms)} tile positions")
        
        # Apply transforms to multichannel data
        print("Applying transforms to multichannel XML...")
        tiles_updated = self._apply_transforms_to_multichannel(
            output_data,
            single_channel_transforms,
            channels_to_apply
        )
        
        print(f"Updated {tiles_updated} tiles in multichannel XML")
        
        # Save the result
        self.save_xml(output_data, output_xml_path)
        print(f"Successfully saved combined XML to: {output_xml_path}")

    def _build_tile_transform_map(
        self,
        data: dict,
        transform_index: int,
    ) -> Dict[str, dict]:
        """
        Build a mapping of tile positions to their selected transform.
        """
        transform_map = {}

        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
        if not isinstance(viewsetups, list):
            viewsetups = [viewsetups]

        for viewsetup in viewsetups:
            tile_id = self.get_tile_id_from_name(data, viewsetup["name"])
            tile_name = self.extract_tile_name_from_viewsetup(viewsetup)

            transforms = self.extract_transforms_for_tile(data, tile_id)

            if len(transforms) <= transform_index:
                print(f"  Warning: No transform at index {transform_index} for {tile_name}")
                continue

            transform = transforms[transform_index]
            transform_name = transform.get("Name", "")

            if (
                "RigidModel3D" in transform_name
                or "AffineModel3D" in transform_name
                or transform_name == "Stitching Transform"
            ):
                transform_map[tile_name] = transform
                print(f"  Stored transform for {tile_name}: {transform_name}")
            else:
                print(
                    f"  Skipped transform for {tile_name}: "
                    f"index={transform_index}, name={transform_name}"
                )

        return transform_map
    
    def _apply_transforms_to_multichannel(
        self,
        data: dict,
        transform_map: Dict[str, dict],
        channels_to_apply: Optional[List[int]] = None
    ) -> int:
        """
        Apply transforms from map to multichannel XML data.
        
        Parameters
        ----------
        data : dict
            Multichannel XML data (modified in place)
        transform_map : dict
            Mapping of tile positions to transforms
        channels_to_apply : list, optional
            Channels to apply transforms to
            
        Returns
        -------
        int
            Number of tiles updated
        """
        tiles_updated = 0
        
        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
        if not isinstance(viewsetups, list):
            viewsetups = [viewsetups]
        
        view_registrations = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
        if not isinstance(view_registrations, list):
            view_registrations = [view_registrations]
            data["SpimData"]["ViewRegistrations"]["ViewRegistration"] = view_registrations
        
        for viewsetup in viewsetups:
            tile_id = self.get_tile_id_from_name(data, viewsetup['name'])
            tile_name = self.extract_tile_name_from_viewsetup(viewsetup)
            channel = self.get_channel_from_viewsetup(viewsetup)
            
            # Check if we should apply to this channel
            if channels_to_apply and channel not in channels_to_apply:
                continue
            
            # Check if we have a transform for this tile position
            if tile_name in transform_map:
                # Find the corresponding ViewRegistration
                for view_reg in view_registrations:
                    if int(view_reg.get("@setup", -1)) == tile_id:
                        # Get existing transforms
                        existing_transforms = view_reg.get("ViewTransform", [])
                        if not isinstance(existing_transforms, list):
                            existing_transforms = [existing_transforms]
                        
                        # Prepend the stitching transform
                        new_transform = copy.deepcopy(transform_map[tile_name])
                        # new_transform["Name"] = f"Stitching Transform from Single Channel"
                        
                        # Add to beginning of transform list
                        existing_transforms.insert(0, new_transform)
                        view_reg["ViewTransform"] = existing_transforms
                        
                        tiles_updated += 1
                        print(f"  Applied transform to {viewsetup.get('name', 'unknown')} (channel {channel})")
                        break
        
        return tiles_updated
    
    def split_multichannel_xml(
        self,
        multichannel_xml_path: str,
        output_dir: str,
        output_prefix: str = "channel"
    ) -> Dict[int, str]:
        """
        Split a multichannel XML into individual channel XMLs.
        
        Parameters
        ----------
        multichannel_xml_path : str
            Path to multichannel XML
        output_dir : str
            Directory for output XMLs
        output_prefix : str
            Prefix for output files (default: "channel")
            
        Returns
        -------
        dict
            Mapping of channel wavelengths to output file paths
        """
        print("Loading multichannel XML...")
        data = self.load_xml(multichannel_xml_path)
        
        # Group ViewSetups by channel
        channel_groups = self._group_viewsetups_by_channel(data)
        
        print(f"Found {len(channel_groups)} channels to split")
        
        output_files = {}
        
        for channel, viewsetup_ids in channel_groups.items():
            print(f"\nProcessing channel {channel} with {len(viewsetup_ids)} tiles...")
            
            # Create a copy for this channel
            channel_data = copy.deepcopy(data)
            
            # Filter ViewSetups and ViewRegistrations
            # self._filter_xml_to_channel(channel_data, viewsetup_ids)
            reindex_tiles = True            
            # Build ID mapping if reindexing
            id_mapping = {}
            if reindex_tiles:
                for new_id, old_id in enumerate(sorted(viewsetup_ids)):
                    id_mapping[old_id] = new_id
                print(f"  Reindexing {len(id_mapping)} tiles starting from 0")
            else:
                # Identity mapping
                id_mapping = {vid: vid for vid in viewsetup_ids}
            self._filter_xml_to_channel_complete(channel_data, viewsetup_ids, id_mapping = id_mapping, channel = int(channel))
            
            # Save the channel-specific XML
            output_path = f"{output_dir}/{output_prefix}_{channel}.xml"
            self.save_xml(channel_data, output_path)
            output_files[channel] = output_path
            
            print(f"  Saved channel {channel} to: {output_path}")
        
        return output_files
    
    def _group_viewsetups_by_channel(self, data: dict) -> Dict[int, List[int]]:
        """
        Group ViewSetup IDs by channel.
        
        Parameters
        ----------
        data : dict
            Parsed XML data
            
        Returns
        -------
        dict
            Mapping of channel wavelengths to lists of ViewSetup IDs
        """
        channel_groups = {}
        
        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
        if not isinstance(viewsetups, list):
            viewsetups = [viewsetups]
        
        for viewsetup in viewsetups:
            tile_id = int(viewsetup.get("id", -1))
            channel = self.get_channel_from_viewsetup(viewsetup)
            
            if channel:
                if channel not in channel_groups:
                    channel_groups[channel] = []
                channel_groups[channel].append(tile_id)
        
        return channel_groups
    
    def _filter_xml_to_channel_complete(
        self, 
        data: dict, 
        viewsetup_ids: List[int],
        id_mapping: Dict[int, int],
        channel: int
    ) -> None:
        """
        Comprehensively filter XML data to only include data for a single channel.
        This includes ViewSetups, ViewRegistrations, ImageLoader paths, and all references.
        
        Parameters
        ----------
        data : dict
            XML data (modified in place)
        viewsetup_ids : list
            List of ViewSetup IDs to keep
        id_mapping : dict
            Mapping from old IDs to new IDs (can be identity mapping)
        channel : int
            Channel wavelength being extracted
        """
        # 1. Filter and optionally reindex ViewSetups
        viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
        if not isinstance(viewsetups, list):
            viewsetups = [viewsetups]
        
        filtered_viewsetups = []
        kept_tile_names = []  # Track which tiles we're keeping
        for vs in viewsetups:
            old_id = int(vs.get("id", -1))
            if old_id in viewsetup_ids:
                # Store the tile name for later filtering
                kept_tile_names.append(vs.get("name", ""))
                
                # Update ID if reindexing
                vs["id"] = str(id_mapping[old_id])
                
                # Update tile attribute if reindexing
                if "attributes" in vs and "tile" in vs["attributes"]:
                    vs["attributes"]["tile"] = str(id_mapping[old_id])
                
                filtered_viewsetups.append(vs)
        
        data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"] = filtered_viewsetups
        
        # 2. Filter and update ViewRegistrations
        view_registrations = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
        if not isinstance(view_registrations, list):
            view_registrations = [view_registrations]
        
        filtered_registrations = []
        for vr in view_registrations:
            old_setup = int(vr.get("@setup", -1))
            if old_setup in viewsetup_ids:
                # Update setup reference
                vr["@setup"] = str(id_mapping[old_setup])
                               
                filtered_registrations.append(vr)
        
        data["SpimData"]["ViewRegistrations"]["ViewRegistration"] = filtered_registrations
        
        # 3. Update ImageLoader section to only include relevant zarr paths
        if "SequenceDescription" in data["SpimData"] and "ImageLoader" in data["SpimData"]["SequenceDescription"]:
            image_loader = data["SpimData"]["SequenceDescription"]["ImageLoader"]
            
            # Handle Zarr format with zgroups
            if "@format" in image_loader and "zarr" in image_loader["@format"].lower():
                # Filter zgroups entries
                if "zgroups" in image_loader:
                    zgroups_data = image_loader["zgroups"]
                    if "zgroup" in zgroups_data:
                        zgroups = zgroups_data["zgroup"]
                        if not isinstance(zgroups, list):
                            zgroups = [zgroups]
                        
                        filtered_zgroups = []
                        for zg in zgroups:
                            # Check if this zgroup's setup is in our filtered list
                            if "@setup" in zg:
                                old_setup = int(zg["@setup"])
                                if old_setup in viewsetup_ids:
                                    # Update setup reference
                                    zg["@setup"] = str(id_mapping[old_setup])
                                    # Also update timepoint if needed
                                    if "@timepoint" in zg:
                                        zg["@timepoint"] = "0"
                                    filtered_zgroups.append(zg)
                        
                        zgroups_data["zgroup"] = filtered_zgroups
                
                # Also handle legacy dataset entries if they exist
                if "zarr" in image_loader:
                    zarr_data = image_loader["zarr"]
                    if "dataset" in zarr_data:
                        datasets = zarr_data["dataset"]
                        if not isinstance(datasets, list):
                            datasets = [datasets]
                        
                        filtered_datasets = []
                        for ds in datasets:
                            if "path" in ds and f"ch_{channel}" in ds["path"]:
                                if "@setup" in ds:
                                    old_setup = int(ds["@setup"])
                                    if old_setup in viewsetup_ids:
                                        ds["@setup"] = str(id_mapping[old_setup])
                                        filtered_datasets.append(ds)
                                else:
                                    filtered_datasets.append(ds)
                        
                        if filtered_datasets:
                            zarr_data["dataset"] = filtered_datasets
                        elif "dataset" in zarr_data:
                            del zarr_data["dataset"]
        
        # 4. Filter ViewInterestPoints if present
        if "ViewInterestPoints" in data["SpimData"]:
            vip = data["SpimData"]["ViewInterestPoints"]
            if "ViewInterestPoint" in vip:
                view_interest_points = vip["ViewInterestPoint"]
                if not isinstance(view_interest_points, list):
                    view_interest_points = [view_interest_points]
                
                filtered_vips = []
                for vip_entry in view_interest_points:
                    old_setup = int(vip_entry.get("@setup", -1))
                    if old_setup in viewsetup_ids:
                        vip_entry["@setup"] = str(id_mapping[old_setup])
                        filtered_vips.append(vip_entry)
                
                if filtered_vips:
                    data["SpimData"]["ViewInterestPoints"]["ViewInterestPoint"] = filtered_vips
                else:
                    # Remove empty ViewInterestPoints section
                    del data["SpimData"]["ViewInterestPoints"]
        
        # 5. Filter Tile Attributes to only include kept tiles
        if "Attributes" in data["SpimData"]["SequenceDescription"]["ViewSetups"]:
            attributes = data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"]
            
            # Filter Tile attributes
            if isinstance(attributes, list):
                for attr_section in attributes:
                    if attr_section.get("@name") == "tile" and "Tile" in attr_section:
                        tiles = attr_section["Tile"]
                        if not isinstance(tiles, list):
                            tiles = [tiles]
                        filtered_tiles = []
                        for tile in tiles:
                            tile_name = tile.get("name", "")
                            original_tile_id = int(tile.get("id", -1))

                            if tile_name in kept_tile_names and original_tile_id in id_mapping:
                                tile["id"] = str(id_mapping[original_tile_id])
                                filtered_tiles.append(tile)
                        attr_section["Tile"] = filtered_tiles
            else:
                # Handle non-list Attributes structure
                if "Tile" in attributes:
                    tiles = attributes["Tile"]
                    if not isinstance(tiles, list):
                        tiles = [tiles]
                    
                    filtered_tiles = []
                    for tile in tiles:
                        tile_name = tile.get("name", "")
                        if tile_name in kept_tile_names:
                            filtered_tiles.append(tile)
                    
                    attributes["Tile"] = filtered_tiles
            
            # Update Channel attributes to only show this channel
            if "Channel" in attributes[1]:
                channels = attributes[1]["Channel"]
                if not isinstance(channels, list):
                    channels = [channels]
                
                # Find the channel entry that matches our wavelength
                matching_channel = None
                for ch in channels:
                    if "name" in ch and str(channel) in ch["name"]:
                        matching_channel = ch
                        break
                    elif "id" in ch and str(ch["id"]) == str(channel):
                        matching_channel = ch
                        break
                
                if matching_channel:
                    # Keep only this channel with simplified ID
                    matching_channel["id"] = str(channel)
                    attributes[1]["Channel"] = matching_channel
        
        # 6. Clean up the total number of setups/timepoints if specified
        if "SequenceDescription" in data["SpimData"]:
            seq_desc = data["SpimData"]["SequenceDescription"]
            
            # Update Timepoints if it lists specific setups
            if "Timepoints" in seq_desc:
                timepoints = seq_desc["Timepoints"]
                if "@type" in timepoints and timepoints["@type"] == "range":
                    # Update range to match filtered setups
                    if "first" in timepoints and "last" in timepoints:
                        new_first = min(id_mapping.values())
                        new_last = max(id_mapping.values())
                        timepoints["first"] = str(new_first)
                        timepoints["last"] = str(new_last)
        
        print(f"    Filtered data to {len(filtered_viewsetups)} tiles for channel {channel}")
        print(f"    Updated {len(filtered_registrations)} view registrations")
        
        # Log what was filtered
        if "ImageLoader" in data["SpimData"]["SequenceDescription"]:
            if "zgroups" in data["SpimData"]["SequenceDescription"]["ImageLoader"]:
                zgroups_data = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"]
                if "zgroup" in zgroups_data:
                    zgroups = zgroups_data["zgroup"]
                    if not isinstance(zgroups, list):
                        zgroups = [zgroups]
                    print(f"    Retained {len(zgroups)} zgroup entries")
            
            # Report on Tile filtering
            if "Attributes" in data["SpimData"]["SequenceDescription"]["ViewSetups"]:
                attributes = data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"]
                if "Tile" in attributes:
                    tiles = attributes["Tile"]
                    if not isinstance(tiles, list):
                        tiles = [tiles]
                    print(f"    Filtered to {len(tiles)} tile attributes")

    def validate_transform_transfer(
        self,
        original_xml_path: str,
        updated_xml_path: str
    ) -> Dict[str, any]:
        """
        Validate that transforms were correctly transferred.
        
        Parameters
        ----------
        original_xml_path : str
            Path to original multichannel XML
        updated_xml_path : str
            Path to updated XML with transferred transforms
            
        Returns
        -------
        dict
            Validation results
        """
        original_data = self.load_xml(original_xml_path)
        updated_data = self.load_xml(updated_xml_path)
        
        original_viewregs = original_data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
        updated_viewregs = updated_data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
        
        if not isinstance(original_viewregs, list):
            original_viewregs = [original_viewregs]
        if not isinstance(updated_viewregs, list):
            updated_viewregs = [updated_viewregs]
        
        results = {
            "total_tiles": len(updated_viewregs),
            "tiles_with_new_transforms": 0,
            "transform_differences": []
        }
        
        for updated_vr in updated_viewregs:
            tile_id = int(updated_vr.get("@setup", -1))
            
            # Find corresponding original
            original_vr = None
            for orig_vr in original_viewregs:
                if int(orig_vr.get("@setup", -1)) == tile_id:
                    original_vr = orig_vr
                    break
            
            if original_vr:
                original_transforms = original_vr.get("ViewTransform", [])
                updated_transforms = updated_vr.get("ViewTransform", [])
                
                if not isinstance(original_transforms, list):
                    original_transforms = [original_transforms]
                if not isinstance(updated_transforms, list):
                    updated_transforms = [updated_transforms]
                
                if len(updated_transforms) > len(original_transforms):
                    results["tiles_with_new_transforms"] += 1
                    results["transform_differences"].append({
                        "tile_id": tile_id,
                        "original_count": len(original_transforms),
                        "updated_count": len(updated_transforms)
                    })
        
        return results

    def merge_single_channel_xmls(
        self,
        single_channel_xml_paths: Dict[int, str],
        output_xml_path: str,
        base_xml_path: Optional[str] = None
    ) -> None:
        """
        Merge multiple single-channel XML files into a multichannel XML.
        
        Parameters
        ----------
        single_channel_xml_paths : dict
            Mapping of channel wavelengths to XML file paths
            e.g., {488: "channel_488.xml", 561: "channel_561.xml", 647: "channel_647.xml"}
        output_xml_path : str
            Path for output multichannel XML
        base_xml_path : str, optional
            Path to a base XML to use as template (uses first channel if not provided)
        """
        print(f"Merging {len(single_channel_xml_paths)} single-channel XMLs into multichannel...")
        
        # Sort channels for consistent ordering
        sorted_channels = sorted(single_channel_xml_paths.keys())
        
        # Load all single-channel XMLs
        channel_data = {}
        for channel in sorted_channels:
            print(f"  Loading channel {channel} from {single_channel_xml_paths[channel]}")
            channel_data[channel] = self.load_xml(single_channel_xml_paths[channel])
        
        # Use base XML or first channel as template
        if base_xml_path:
            print(f"Using base XML template: {base_xml_path}")
            merged_data = self.load_xml(base_xml_path)
        else:
            first_channel = sorted_channels[0]
            print(f"Using channel {first_channel} as template")
            merged_data = copy.deepcopy(channel_data[first_channel])
        
        # Build ID mapping for each channel
        id_offset = 0
        channel_id_mappings = {}
        
        for channel in sorted_channels:
            data = channel_data[channel]
            viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
            if not isinstance(viewsetups, list):
                viewsetups = [viewsetups]
            
            # Create mapping from old IDs to new sequential IDs
            old_ids = sorted([int(vs.get("id", -1)) for vs in viewsetups])
            id_mapping = {}
            for old_id in old_ids:
                id_mapping[old_id] = id_offset
                id_offset += 1
            
            channel_id_mappings[channel] = id_mapping
            print(f"  Channel {channel}: {len(id_mapping)} tiles, IDs {min(id_mapping.values())}-{max(id_mapping.values())}")
        
        # Merge ViewSetups
        print("\nMerging ViewSetups...")
        merged_viewsetups = []
        
        for channel in sorted_channels:
            data = channel_data[channel]
            id_mapping = channel_id_mappings[channel]
            
            viewsetups = data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"]
            if not isinstance(viewsetups, list):
                viewsetups = [viewsetups]
            
            for vs in viewsetups:
                # Create a copy to avoid modifying original
                vs_copy = copy.deepcopy(vs)
                
                # Update IDs
                old_id = int(vs_copy.get("id", -1))
                new_id = id_mapping[old_id]
                vs_copy["id"] = str(new_id)
                
                # Update tile attribute
                if "attributes" in vs_copy and "tile" in vs_copy["attributes"]:
                    vs_copy["attributes"]["tile"] = str(new_id)
                
                # Ensure channel attribute is set correctly
                if "attributes" in vs_copy:
                    vs_copy["attributes"]["channel"] = str(channel)
                
                # Update name to include channel if not already present
                name = vs_copy.get("name", "")
                if f"ch_{channel}" not in name and f"CH_{channel}" not in name:
                    # Extract tile position and add channel
                    tile_pattern = r"(Tile_X_\d+_Y_\d+_Z_\d+)"
                    match = re.search(tile_pattern, name)
                    if match:
                        tile_name = match.group(1)
                        vs_copy["name"] = f"{tile_name}_ch_{channel}.ome.zarr"
                    else:
                        vs_copy["name"] = f"{name}_ch_{channel}"
                
                merged_viewsetups.append(vs_copy)
        
        merged_data["SpimData"]["SequenceDescription"]["ViewSetups"]["ViewSetup"] = merged_viewsetups
        print(f"  Merged {len(merged_viewsetups)} total ViewSetups")
        
        # Merge ViewRegistrations
        print("\nMerging ViewRegistrations...")
        merged_registrations = []
        
        for channel in sorted_channels:
            data = channel_data[channel]
            id_mapping = channel_id_mappings[channel]
            
            view_registrations = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
            if not isinstance(view_registrations, list):
                view_registrations = [view_registrations]
            
            for vr in view_registrations:
                # Create a copy
                vr_copy = copy.deepcopy(vr)
                
                # Update setup reference
                old_setup = int(vr_copy.get("@setup", -1))
                if old_setup in id_mapping:
                    vr_copy["@setup"] = str(id_mapping[old_setup])
                    merged_registrations.append(vr_copy)
        
        merged_data["SpimData"]["ViewRegistrations"]["ViewRegistration"] = merged_registrations
        print(f"  Merged {len(merged_registrations)} total ViewRegistrations")
        
        # Merge ImageLoader paths
        print("\nMerging ImageLoader paths...")
        self._merge_image_loader_paths(merged_data, channel_data, channel_id_mappings, sorted_channels)
        
        # Merge Attributes
        print("\nMerging Attributes...")
        self._merge_attributes(merged_data, channel_data, channel_id_mappings, sorted_channels)
        
        # Merge ViewInterestPoints if present
        self._merge_view_interest_points(merged_data, channel_data, channel_id_mappings, sorted_channels)
        
        # Save the merged XML
        self.save_xml(merged_data, output_xml_path)
        print(f"\nSuccessfully merged {len(sorted_channels)} channels into: {output_xml_path}")

    def _merge_image_loader_paths(
        self,
        merged_data: dict,
        channel_data: Dict[int, dict],
        channel_id_mappings: Dict[int, Dict[int, int]],
        sorted_channels: List[int]
    ) -> None:
        """
        Merge ImageLoader paths from all channels.
        """
        image_loader = merged_data["SpimData"]["SequenceDescription"].get("ImageLoader", {})
        
        # Handle Zarr format with zgroups
        if "@format" in image_loader and "zarr" in image_loader["@format"].lower():
            merged_zgroups = []
            
            for channel in sorted_channels:
                data = channel_data[channel]
                id_mapping = channel_id_mappings[channel]
                
                if "ImageLoader" in data["SpimData"]["SequenceDescription"]:
                    channel_loader = data["SpimData"]["SequenceDescription"]["ImageLoader"]
                    
                    if "zgroups" in channel_loader and "zgroup" in channel_loader["zgroups"]:
                        zgroups = channel_loader["zgroups"]["zgroup"]
                        if not isinstance(zgroups, list):
                            zgroups = [zgroups]
                        
                        for zg in zgroups:
                            zg_copy = copy.deepcopy(zg)
                            
                            # Update setup reference
                            if "@setup" in zg_copy:
                                old_setup = int(zg_copy["@setup"])
                                if old_setup in id_mapping:
                                    zg_copy["@setup"] = str(id_mapping[old_setup])
                            
                            # Ensure path includes channel information
                            path = zg_copy.get("path", "")
                            if f"ch_{channel}" not in path and f"CH_{channel}" not in path:
                                # Add channel to path if not present
                                if path.endswith(".ome.zarr"):
                                    base_path = path[:-9]  # Remove .ome.zarr
                                    zg_copy["path"] = f"{base_path}_ch_{channel}.ome.zarr"
                                else:
                                    zg_copy["path"] = f"{path}_ch_{channel}"
                            
                            merged_zgroups.append(zg_copy)
            
            # Update merged data
            if "zgroups" not in image_loader:
                image_loader["zgroups"] = {}
            image_loader["zgroups"]["zgroup"] = merged_zgroups
            print(f"  Merged {len(merged_zgroups)} zgroup entries")

    def _merge_attributes(
        self,
        merged_data: dict,
        channel_data: Dict[int, dict],
        channel_id_mappings: Dict[int, Dict[int, int]],
        sorted_channels: List[int]
    ) -> None:
        """
        Merge Attributes sections from all channels.
        """
        # Get or create Attributes section
        if "Attributes" not in merged_data["SpimData"]["SequenceDescription"]["ViewSetups"]:
            merged_data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"] = []
        
        attributes = merged_data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"]
        if not isinstance(attributes, list):
            attributes = [attributes]
        
        # Find or create each attribute type
        tile_attr = None
        channel_attr = None
        
        for attr in attributes:
            if attr.get("@name") == "tile":
                tile_attr = attr
            elif attr.get("@name") == "channel":
                channel_attr = attr
        
        # Merge Tiles
        merged_tiles = []
        tile_names_seen = set()
        
        for channel in sorted_channels:
            data = channel_data[channel]
            id_mapping = channel_id_mappings[channel]
            
            if "Attributes" in data["SpimData"]["SequenceDescription"]["ViewSetups"]:
                attrs = data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"]
                if not isinstance(attrs, list):
                    attrs = [attrs]
                
                for attr in attrs:
                    if attr.get("@name") == "tile" or "Tile" in attr:
                        tiles = attr.get("Tile", [])
                        if not isinstance(tiles, list):
                            tiles = [tiles]
                        
                        for tile in tiles:
                            tile_copy = copy.deepcopy(tile)
                            
                            # Update tile ID
                            old_id = int(tile.get("id", -1))
                            if old_id in id_mapping:
                                tile_copy["id"] = str(id_mapping[old_id])
                                
                                # Update name to include channel if needed
                                name = tile_copy.get("name", "")
                                if f"ch_{channel}" not in name and f"CH_{channel}" not in name:
                                    tile_pattern = r"(Tile_X_\d+_Y_\d+_Z_\d+)"
                                    match = re.search(tile_pattern, name)
                                    if match:
                                        tile_pos = match.group(1)
                                        tile_copy["name"] = f"{tile_pos}_ch_{channel}.ome.zarr"
                                
                                # Only add if we haven't seen this exact tile position/channel combo
                                tile_key = (tile_copy.get("name", ""), channel)
                                if tile_key not in tile_names_seen:
                                    merged_tiles.append(tile_copy)
                                    tile_names_seen.add(tile_key)
        
        # Update or create tile attribute
        if tile_attr:
            tile_attr["Tile"] = merged_tiles
        else:
            attributes.append({
                "@name": "tile",
                "Tile": merged_tiles
            })
        
        # Merge Channels
        merged_channels = []
        for channel in sorted_channels:
            merged_channels.append({
                "id": str(channel),
                "name": str(channel)
            })
        
        # Update or create channel attribute
        if channel_attr:
            channel_attr["Channel"] = merged_channels
        else:
            attributes.append({
                "@name": "channel",
                "Channel": merged_channels
            })
        
        merged_data["SpimData"]["SequenceDescription"]["ViewSetups"]["Attributes"] = attributes
        print(f"  Merged {len(merged_tiles)} tiles and {len(merged_channels)} channels in Attributes")

    def _merge_view_interest_points(
        self,
        merged_data: dict,
        channel_data: Dict[int, dict],
        channel_id_mappings: Dict[int, Dict[int, int]],
        sorted_channels: List[int]
    ) -> None:
        """
        Merge ViewInterestPoints if present in any channel.
        """
        merged_vips = []
        
        for channel in sorted_channels:
            data = channel_data[channel]
            id_mapping = channel_id_mappings[channel]
            
            if "ViewInterestPoints" in data["SpimData"]:
                vip = data["SpimData"]["ViewInterestPoints"]
                
                # Handle both ViewInterestPoint and ViewInterestPointsFile formats
                if "ViewInterestPoint" in vip:
                    view_interest_points = vip["ViewInterestPoint"]
                    if not isinstance(view_interest_points, list):
                        view_interest_points = [view_interest_points]
                    
                    for vip_entry in view_interest_points:
                        vip_copy = copy.deepcopy(vip_entry)
                        old_setup = int(vip_copy.get("@setup", -1))
                        if old_setup in id_mapping:
                            vip_copy["@setup"] = str(id_mapping[old_setup])
                            merged_vips.append(vip_copy)
                
                elif "ViewInterestPointsFile" in vip:
                    vip_files = vip["ViewInterestPointsFile"]
                    if not isinstance(vip_files, list):
                        vip_files = [vip_files]
                    
                    for vip_file in vip_files:
                        vip_copy = copy.deepcopy(vip_file)
                        if "@setup" in vip_copy:
                            old_setup = int(vip_copy["@setup"])
                            if old_setup in id_mapping:
                                vip_copy["@setup"] = str(id_mapping[old_setup])
                                
                                # Update the file path to reflect new ID
                                if "#text" in vip_copy:
                                    old_path = vip_copy["#text"]
                                    # Replace old viewSetupId with new one
                                    new_path = re.sub(
                                        f"viewSetupId_{old_setup}",
                                        f"viewSetupId_{id_mapping[old_setup]}",
                                        old_path
                                    )
                                    vip_copy["#text"] = new_path
                                
                                merged_vips.append(vip_copy)
        
        # Add merged ViewInterestPoints if any exist
        if merged_vips:
            if "ViewInterestPoints" not in merged_data["SpimData"]:
                merged_data["SpimData"]["ViewInterestPoints"] = {}
            
            # Determine format based on first entry
            if merged_vips and "@setup" in merged_vips[0]:
                merged_data["SpimData"]["ViewInterestPoints"]["ViewInterestPointsFile"] = merged_vips
            else:
                merged_data["SpimData"]["ViewInterestPoints"]["ViewInterestPoint"] = merged_vips
            
            print(f"  Merged {len(merged_vips)} ViewInterestPoints")


def discover_channel_xmls(
    data_dir: str = "/data/",
    xml_filename: str = "rhapso-solver-affine.xml"
) -> Dict[int, str]:
    """
    Discover channel XML files by scanning directory structure.
    
    Looks for folders matching the pattern 'ch_{wavelength}' and builds
    a dictionary mapping wavelengths to XML file paths.
    
    Parameters
    ----------
    data_dir : str
        Root directory to search in (default: "/data/")
    xml_filename : str
        Name of the XML file in each channel folder (default: "rhapso-solver-affine.xml")

    
    Returns
    -------
    dict
        Mapping of channel wavelengths (int) to XML file paths (str)
        e.g., {488: "/data/ch_488/rhapso-solver-affine.xml", ...}
    
    Examples
    --------
    >>> # Basic usage
    >>> channel_xmls = discover_channel_xmls()
    >>> print(channel_xmls)
    {488: '/data/ch_488/rhapso-solver-affine.xml', 
     561: '/data/ch_561/rhapso-solver-affine.xml',
     647: '/data/ch_647/rhapso-solver-affine.xml'}
    
    >>> # Then use with merge function
    >>> merge_single_channel_xmls_to_multichannel(
    ...     channel_xml_paths=channel_xmls,
    ...     output_xml="/scratch/merged_multichannel.xml"
    ... )
    """
    channel_pattern: str = r"^ch_(\d+)$"
    channel_xml_paths = {}
    
    # Convert to Path object for easier manipulation
    data_path = Path(data_dir)
    
    # Check if data directory exists
    if not data_path.exists():
        print(f"Warning: Directory {data_dir} does not exist")
        return channel_xml_paths
    
    if not data_path.is_dir():
        print(f"Warning: {data_dir} is not a directory")
        return channel_xml_paths
    
    # Compile the pattern for efficiency
    pattern = re.compile(channel_pattern)
    
    # Scan for channel directories
    print(f"Scanning {data_dir} for channel folders...")
    
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            # Check if folder name matches channel pattern
            match = pattern.match(item.name)
            if match:
                # Extract wavelength from folder name
                wavelength = int(match.group(1))
                
                # Check if XML file exists in this folder
                xml_path = item / xml_filename
                
                if xml_path.exists():
                    channel_xml_paths[wavelength] = str(xml_path)
                    print(f"  Found channel {wavelength}: {xml_path}")
                else:
                    print(f"  Warning: No {xml_filename} found in {item}")
    
    # Summary
    if channel_xml_paths:
        channels = sorted(channel_xml_paths.keys())
        print(f"\nDiscovered {len(channels)} channels: {channels}")
    else:
        print(f"\nNo channel folders found matching pattern '{channel_pattern}' in {data_dir}")
    
    return channel_xml_paths


# Convenience function for merging
def merge_single_channel_xmls_to_multichannel(
    channel_xml_paths: Dict[int, str],
    output_xml: str,
    base_xml: Optional[str] = None
):
    """
    Convenience function to merge single-channel XMLs into multichannel.
    
    Parameters
    ----------
    channel_xml_paths : dict
        Mapping of channel wavelengths to XML paths
        e.g., {488: "channel_488.xml", 561: "channel_561.xml"}
    output_xml : str
        Output path for merged multichannel XML
    base_xml : str, optional
        Optional base XML to use as template
    """
    manager = BigStitcherXMLManager()
    manager.merge_single_channel_xmls(
        channel_xml_paths,
        output_xml,
        base_xml
    )

# Convenience functions for common operations
def transfer_stitching_to_multichannel(
    single_channel_xml: str,
    multichannel_xml: str,
    output_xml: str,
    channels: Optional[List[int]] = None
):
    """
    Convenience function to transfer stitching transforms.
    
    Parameters
    ----------
    single_channel_xml : str
        Path to single-channel stitched XML
    multichannel_xml : str
        Path to multichannel camera-aligned XML
    output_xml : str
        Output path for combined XML
    channels : list, optional
        Channels to apply transforms to
    """
    manager = BigStitcherXMLManager()
    manager.transfer_stitching_transforms(
        single_channel_xml,
        multichannel_xml,
        output_xml,
        transform_index=0,  # Second transform (stitching)
        channels_to_apply=channels
    )


def split_multichannel_xml(xml_path: str, output_dir: str):
    """
    Convenience function to split multichannel XML.
    
    Parameters
    ----------
    xml_path : str
        Path to multichannel XML
    output_dir : str
        Directory for output files
        
    Returns
    -------
    dict
        Mapping of channels to output files
    """
    manager = BigStitcherXMLManager()
    return manager.split_multichannel_xml(xml_path, output_dir)

    


# Example usage
if __name__ == "__main__":
    # # Example 1: Transfer stitching transforms
    # print("=" * 60)
    # print("Example 1: Transfer stitching transforms")
    # print("=" * 60)
    
    # #/root/capsule/data/HCR_000000-s49_2025-08-13_13-00-00_processed_2025-09-10_22-57-56
    # transfer_stitching_to_multichannel(
    #     single_channel_xml="s3://aind-open-data/HCR_000000-s49_2025-08-13_13-00-00_processed_2025-09-10_22-57-56/image_tile_alignment/bigstitcher.xml",
    #     multichannel_xml="s3://aind-open-data/HCR_000000-s49_2025-08-13_13-00-00_processed_2025-09-10_22-57-56/image_tile_alignment/stitching_cam_alignment_spot_channels.xml",
    #     output_xml="/scratch/multichannel_with_stitching.xml",
    #     # channels=[488, 561, 647]  # Optional: only apply to specific channels
    # )
    
    # # Example 2: Split multichannel XML
    # print("\n" + "=" * 60)
    # print("Example 2: Split multichannel XML")
    # print("=" * 60)
    
    # output_files = split_multichannel_xml(
    #     xml_path="/scratch/multichannel_with_stitching.xml",
    #     output_dir="/scratch/single_channel_xmls"
    # )
    # print(f"Created {len(output_files)} channel-specific XMLs")

    # merge_single_channel_xmls_to_multichannel(
    channel_xml_paths={
            405: "/root/capsule/data/ch_405/rhapso-solver-affine.xml",
            488: "/root/capsule/data/ch_488/rhapso-solver-affine.xml", 
            514: "/root/capsule/data/ch_514/rhapso-solver-affine.xml",
            561: "/root/capsule/data/ch_561/rhapso-solver-affine.xml",
            594: "/root/capsule/data/ch_594/rhapso-solver-affine.xml", 
            638: "/root/capsule/data/ch_638/rhapso-solver-affine.xml"
    }
        # output_xml="/scratch/merged_multichannel.xml"
    # )
    
    # Example 3: Validate transform transfer
    # print("\n" + "=" * 60)
    # print("Example 3: Validate transform transfer")
    # print("=" * 60)
    
    # manager = BigStitcherXMLManager()
    # validation = manager.validate_transform_transfer(
    #     original_xml_path="multichannel_aligned.xml",
    #     updated_xml_path="multichannel_with_stitching.xml"
    # )
    # print(f"Validation results:")
    # print(f"  Total tiles: {validation['total_tiles']}")
    # print(f"  Tiles with new transforms: {validation['tiles_with_new_transforms']}")
    data_folder = '/data'
    scratch_folder = '/scratch'
    results_folder = '/results'
    single_stitching_xml_path = "/root/capsule/data/rhapso-solver-affine (23).xml"
    multichannel_xml_path =     f"/root/capsule/data/stitching_cam_alignment_spot_channels (8).xml"
    temp_xml_path =             f"{scratch_folder}/combined_camera_aligned_rhapso_channel_average.xml"
    output_xml_path =           f"{results_folder}/combined_camera_aligned_rhapso_channel_average.xml"

    manager = BigStitcherXMLManager()

    manager.transfer_stitching_transforms(
        single_stitching_xml_path,
        multichannel_xml_path,
        temp_xml_path,
        transform_index=1,  # First transform (rigid)
        channels_to_apply=None)

    manager.transfer_stitching_transforms(
        single_stitching_xml_path,
        temp_xml_path,
        output_xml_path,
        transform_index=0,  # Second transform (affine)
        channels_to_apply=None
    )