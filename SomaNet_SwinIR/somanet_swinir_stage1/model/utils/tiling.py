import numpy as np

class TileImage:
    def __init__(self, tile_size, overlap_size, corner_cut):
        assert len(tile_size) == len(overlap_size) == len(corner_cut), \
            "Tile size, overlap size, and corner cut must have the same number of dimensions."
        
        self.tile_size = tile_size
        self.overlap_size = overlap_size
        self.corner_cut = corner_cut
        self.tiles = []
        self.indices = []

    def tile_image(self, image):
        self.image_shape = image.shape
        self.tiles = []
        self.indices = []
        if len(image.shape) == 2:
            self._tile_image_2d(image)
        elif len(image.shape) == 3:
            self._tile_image_3d(image)
        else:
            raise ValueError("Unsupported image dimensions. Only 2D and 3D images are supported.")
    
    def reassemble_image(self, processed_tiles):
        if len(self.image_shape) == 2:
            return self._reassemble_image_2d(processed_tiles)
        elif len(self.image_shape) == 3:
            return self._reassemble_image_3d(processed_tiles)
        else:
            raise ValueError("Unsupported image dimensions. Only 2D and 3D images are supported.")

    def _tile_image_2d(self, image):
        width, height = image.shape
        tile_width, tile_height = self.tile_size
        overlap_width, overlap_height = self.overlap_size

        for w in range(0, width, tile_width - overlap_width):
            for h in range(0, height, tile_height - overlap_height):
                tile = image[w:min(w+tile_width, width), h:min(h+tile_height, height)]
                self.tiles.append(tile)
                self.indices.append((w, h))

    def _tile_image_3d(self, image):
        channels, width, height = image.shape
        tile_channels, tile_width, tile_height = self.tile_size
        overlap_channels, overlap_width, overlap_height = self.overlap_size

        for c in range(0, channels, tile_channels-overlap_channels):
            for w in range(0, width, tile_width - overlap_width):
                for h in range(0, height, tile_height - overlap_height):
                    tile = image[c:min(c+tile_channels, channels), w:min(w+tile_width, width), h:min(h+tile_height, height)]
                    self.tiles.append(tile)
                    self.indices.append((c, w, h))

    def _reassemble_image_2d(self, processed_tiles):
        image = np.zeros(self.image_shape)
        count_map = np.zeros(self.image_shape)
        cut_width, cut_height = self.corner_cut

        for tile, (w, h) in zip(processed_tiles, self.indices):
            tile_width, tile_height = tile.shape
            if w == 0 or h == 0 or (w + tile_width) >= self.image_shape[0] or (h + tile_height) >= self.image_shape[1]:
                image[w:w+tile_width, h:h+tile_height] = tile
                count_map[w:w+tile_width, h:h+tile_height] = 1
            else:
                cropped_tile = tile[cut_width:-cut_width or None, cut_height:-cut_height or None]
                image[w+cut_width:w+cut_width+cropped_tile.shape[0], h+cut_height:h+cut_height+cropped_tile.shape[1]] += cropped_tile
                count_map[w+cut_width:w+cut_width+cropped_tile.shape[0], h+cut_height:h+cut_height+cropped_tile.shape[1]] += 1

        image /= np.maximum(count_map, 1)  # Avoid division by zero
        return image

    def _reassemble_image_3d(self, processed_tiles):
        image = np.zeros(self.image_shape)
        count_map = np.zeros(self.image_shape)
        cut_channels, cut_width, cut_height = self.corner_cut

        for tile, (c, w, h) in zip(processed_tiles, self.indices):
            tile_channels, tile_width, tile_height = tile.shape
            if (c == 0 or w == 0 or h == 0 or
                (c + tile_channels) >= self.image_shape[0] or
                (w + tile_width) >= self.image_shape[1] or
                (h + tile_height) >= self.image_shape[2]):
                image[c:c+tile_channels, w:w+tile_width, h:h+tile_height] = tile
                count_map[c:c+tile_channels, w:w+tile_width, h:h+tile_height] = 1
            else:
                cropped_tile = tile[
                    cut_channels:-cut_channels or None,
                    cut_width:-cut_width or None,
                    cut_height:-cut_height or None
                ]
                image[
                    c+cut_channels:c+cut_channels+cropped_tile.shape[0],
                    w+cut_width:w+cut_width+cropped_tile.shape[1],
                    h+cut_height:h+cut_height+cropped_tile.shape[2]
                ] += cropped_tile
                count_map[
                    c+cut_channels:c+cut_channels+cropped_tile.shape[0],
                    w+cut_width:w+cut_width+cropped_tile.shape[1],
                    h+cut_height:h+cut_height+cropped_tile.shape[2]
                ] += 1

        image /= np.maximum(count_map, 1)  # Avoid division by zero
        return image


