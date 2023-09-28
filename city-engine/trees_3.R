library(sf)
library(dplyr)
library(magrittr)

n_trees <- 7000

streets <- st_read("C:/Users/katerndaf98/Documents/CityEngine/Default Workspace/bachelor/data/experiment_3_street_network.shp", quiet = TRUE)

streets <- streets %>% 
  mutate(total_segment_width = strtWdth + sdwlkWdthR + sdwlkWdthL)

# keep a distance of 80% of sidewalk width to buildings
outer_edge <- st_buffer(streets, streets$total_segment_width / 2 - 0.8 * streets$sdwlkWdthL) %>% 
  st_union() %>% 
  st_as_sf()

# keep a distance of 0.4 meter to street
inner_edge <- st_buffer(streets, streets$strtWdth / 2 + 0.4) %>%
  st_union() %>%
  st_as_sf()

sidewalks <- st_difference(outer_edge, inner_edge) %>% 
  st_cast("POLYGON")

tree_locations <- st_sample(sidewalks, size = n_trees, type = "random", exact = FALSE) %>% 
  st_as_sf() %>% 
  rename(geometry = x) %>% 
  mutate(type = "tree", .before = geometry)

# remove trees that are closer to each other than x meter; only keep the first entry from a cluster (to be implemented)
# tree_distances <- st_distance(tree_locations, tree_locations)
# self_tree_distances <- tree_distances == 0
# tree_distances[self_tree_distances] <- NA

# write out tree coordinates (x, y, z) as well as rotation around y axis
# NOTE THE * -1!!!
tree_coords <- st_coordinates(tree_locations) %>% 
	as.data.frame() %>% 
	mutate(species = "",
		   X = round(X, 3),
		   Z = round(Y * -1, 3),
		   Y = 0,
		   rot_y = round(runif(n(), 0, 359), 0),
		   scale = rnorm(n(), 0.19, 0.01)) %>% # laubwerk scale: rnorm(n(), 0.01, 0.001); LumenRT(?) scale: rnorm(n(), 0.19, 0.01)
	relocate(species, .before = X)


# WARNING: For testing purposes, I only export `n_trees` trees
write.table(tree_coords, "C:/Users/katerndaf98/Documents/CityEngine/Default Workspace/bachelor/data/experiment_5_tree_attributes.txt",
			sep = ";", eol = "\n", append = FALSE, quote = FALSE, na = "", row.names = FALSE,
			col.names = FALSE, fileEncoding = "ASCII")
# 
# tree_locations %>%
# 	st_write(dsn = "C:/Users/flori/Desktop/Uni/Bachelorarbeit/city-engine/data/treeLOCS.shp", delete_dsn = TRUE)
