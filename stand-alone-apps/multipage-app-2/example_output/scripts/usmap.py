import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def draw_us_map():
    # Create a figure
    plt.figure(figsize=(16, 8))

    # Create a Basemap instance for the continental US
    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

    # Draw state boundaries
    m.drawstates()

    # Draw country boundaries
    m.drawcountries()

    # Draw coastlines
    m.drawcoastlines()

    # Display the map
    plt.show()

if __name__ == "__main__":
    draw_us_map()