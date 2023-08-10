sys.path.append(ce.toFSPath("scripts"))

from Scene import Scene

scene = Scene(ce)

def experiment_7():
    """
    Complete Street, Laubwerk models, no geometry variation
    """
    scene.generate_street_network(True, street_layer_name="streets", number_of_streets=7000, sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.generate_street_network(False, street_layer_name="streets", number_of_streets=1000, major_pattern="RADIAL", minor_pattern="ORGANIC", sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.generate_street_network(False, street_layer_name="streets", number_of_streets=900, force_outwards_growth=False, major_pattern="ORGANIC", minor_pattern="ORGANIC", sidewalk_minimum_width=4, sidewalk_maximum_width=6, lanes_maximum=8)
    
    scene.diversify_city_blocks()
    
    scene.set_rule_files({"park": "rules/parks.cga", "street": "rules/Complete_Street.cga", "lot": {"file": "/ESRI.lib/rules/Buildings/Building_From_Footprint.cga", "start": "Generate"}})
    
    vp = scene.get_current_viewport()
    
    scene.export_as_shape_file("streets", "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\data\\experiment_7_street_network.shp")
    
    _ = input("run R script\n")
    
    scene.place_street_trees("C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\models\\laubwerk-trees\\**\\*.obj", "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\data\\experiment_7_tree_attributes.txt")
    
    scene.place_park_trees(0.0004, 0.0008, 0.0008, "trees", "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\models\\laubwerk-trees\\**\\*.obj", scale_min=0.019, scale_max=0.011)
    
    scene.gather_tree_images("trees", vp, "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\images\\experiment_7", "city_trees.png", 512, "C:\\Users\\flori\\Desktop\\Uni\\Bachelorarbeit\\city-engine\\images\\experiment_7\\meta.csv", mean_height=120.0, mean_height_sd=0.0,
                            lighting_settings={"light_month": 6, "light_time_zone": 1, "shadow_quality": "SHADOW_HIGH", "ambient_occlusion_samples": "AMBIENT_OCCLUSION_SAMPLES_HIGHEST", "sun_source": "SUN_POSITION_SOURCE_TIME_DATE"},
                            camera_settings={"_randomize": False},
                            render_settings=({"axes_visible": False, "grid_visible": False}, {}),
                            truth_detection_strategy="diff",
                            position_noise=False, rotation_noise=False)


