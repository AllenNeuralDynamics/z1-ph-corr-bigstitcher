from xml_utils import transfer_stitching_to_multichannel, split_multichannel_xml

def combine_xmls():
    output_big_stitcher_xml = '../results/bigstitcher.xml'
    # output_big_stitcher_xml = '/results/bigstitcher.xml'
    CAMERA_ALIGNED_XML_PATH =  "../data/stitching_cam_alignment_spot_channels.xml"
    # CAMERA_ALIGNED_XML_PATH =  "/data/stitching_cam_alignment_forward_transform_spot_channels.xml"

    COMBINED_XML_PATH = "../results/combined_stitching_cam_alignment_all_channels.xml"
    COMBINED_XML_PATH_SCRATCH = "../scratch/combined_stitching_cam_alignment_all_channels.xml"

    try: 
        transfer_stitching_to_multichannel(single_channel_xml = output_big_stitcher_xml, 
        multichannel_xml = CAMERA_ALIGNED_XML_PATH, 
        output_xml = COMBINED_XML_PATH)

        transfer_stitching_to_multichannel(single_channel_xml = output_big_stitcher_xml, 
        multichannel_xml = CAMERA_ALIGNED_XML_PATH, 
        output_xml = COMBINED_XML_PATH_SCRATCH)
    # except: 
    #     print(f'Error combining xmls')
    except Exception as e:
        print("Error combining xmls")
        print(f"{type(e).__name__}: {e}")
        raise

    
def split_xmls_to_single_channel_xmls():
    COMBINED_XML_PATH_SCRATCH = "../scratch/combined_stitching_cam_alignment_all_channels.xml"
    try:
        output_files = split_multichannel_xml(
            xml_path=COMBINED_XML_PATH_SCRATCH,
            output_dir="../results/single_channel_xmls"
        )
        print(f"Created {len(output_files)} channel-specific XMLs")
    except: 
        print(f'Error splitting xmls')


if __name__ == "__main__": 
    combine_xmls()
    split_xmls_to_single_channel_xmls()