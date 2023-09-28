sys.path.append(ce.toFSPath("scripts"))

from Scene import Scene

scene = Scene(ce)

def sub_8():
    """
    Complete Street, Laubwerk models, geometry variation
    """
    vp = scene.get_current_viewport()
    
    scene.gather_tree_images("trees", vp,  "D:\\Florian_Katerndahl\\bachelor\\images\\experiment_8", "city_trees.png", 512, "D:\\Florian_Katerndahl\\bachelor\\images\\experiment_8\\meta.csv", mean_height=120.0, mean_height_sd=0.0,
                            lighting_settings={"light_month": 6, "light_time_zone": 1, "shadow_quality": "SHADOW_HIGH", "ambient_occlusion_samples": "AMBIENT_OCCLUSION_SAMPLES_HIGHEST", "sun_source": "SUN_POSITION_SOURCE_TIME_DATE"},
                            camera_settings={"_randomize": True, "angle_sd": 15.0},
                            render_settings=({"axes_visible": False, "grid_visible": False}, {}),
                            truth_detection_strategy="diff",
                            position_noise=False, rotation_noise=True, rotation_sd=10.0)
 

def experiment_8():
    """
    Complete Street, Laubwerk models, geometry variation
    """
    scene.generate_street_network(True, street_layer_name="streets", number_of_streets=1000, sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.generate_street_network(False, street_layer_name="streets", number_of_streets=3000, major_pattern="RADIAL", minor_pattern="ORGANIC", sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.generate_street_network(False, street_layer_name="streets", number_of_streets=900, force_outwards_growth=False, major_pattern="ORGANIC", minor_pattern="ORGANIC", sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.diversify_city_blocks()
    
    scene.set_rule_files({"park": "rules/parks.cga", "street": "rules/CompleteStreet-customized.cga", "lot": {"file": "/ESRI.lib/rules/Buildings/Building_From_Footprint.cga", "start": "Generate"}})
    
    #vp = scene.get_current_viewport()
    
    scene.export_as_shape_file("streets", "D:\\Florian_Katerndahl\\bachelor\\data\\experiment_8_street_network.shp")
    
    _ = input("run R script\n")
    
    scene.place_street_trees("D:\\Florian_Katerndahl\\bachelor\\assets\\Plants-subset\\**\\*.obj", "D:\\Florian_Katerndahl\\bachelor\\data\\experiment_8_tree_attributes.txt")
    
    scene.place_park_trees(0.0004, 0.0008, 0.0008, "trees", "D:\\Florian_Katerndahl\\bachelor\\assets\\Plants-subset\\**\\*.obj", scale_min=0.019, scale_max=0.021)
    
    #scene.gather_tree_images("trees", vp, "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\images\\experiment_8", "city_trees.png", 512, "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\images\\experiment_8\\meta.csv", mean_height=120.0, mean_height_sd=0.0,
    #                        lighting_settings={"light_month": 6, "light_time_zone": 1, "shadow_quality": "SHADOW_HIGH", "ambient_occlusion_samples": "AMBIENT_OCCLUSION_SAMPLES_HIGHEST", "sun_source": "SUN_POSITION_SOURCE_TIME_DATE"},
    #                        camera_settings={"_randomize": True, "angle_sd": 15.0},
    #                        render_settings=({"axes_visible": False, "grid_visible": False}, {}),
    #                        truth_detection_strategy="diff",
    #                        position_noise=False, rotation_noise=True, rotation_sd=10.0)


