import numpy as np
import bokeh.palettes as bokeh_palettes

# This file exists because
# (1) I have issues with some Bokeh palettes, namely that RdYlGn is reversed from the sensible Matplotlib convention
# (2) I wanted some Bokeh palettes which were not yet available in the version I had to use
# TODO: Reason (2) at least is no longer relevant... so I can probably reduce this file dramatically


# Scheme from https://personal.sron.nl/~pault/#fig:scheme_rainbow_discrete_all
TolRainbow23 = (
    '#E8ECFB', '#D9CCE3', '#CAACCB', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#6195CF',
    '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F4A736', '#EE8026', '#E65518', '#DC050C',
    '#A5170E', '#72190E', '#42150A')
#TolRainbow22 = (
#    '#D9CCE3', '#CAACCB', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#6195CF', '#7BAFDE',
#    '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F4A736', '#EE8026', '#E65518', '#DC050C', '#A5170E',
#    '#72190E', '#42150A')
#TolRainbow21 = (
#    '#D9CCE3', '#CAACCB', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#6195CF', '#7BAFDE',
#    '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F7CB45', '#F4A736', '#EE8026', '#E65518', '#DC050C', '#A5170E', '#72190E')
#TolRainbow20 = (
#    '#D9CCE3', '#CAACCB', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#437DBF', '#6195CF', '#7BAFDE',
#    '#4EB265', '#90C987', '#CAE0AB', '#F7F056', '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#A5170E', '#72190E')
#TolRainbow19 = (
#    '#D9CCE3', '#CAACCB', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265',
#    '#90C987', '#CAE0AB', '#F7F056', '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#A5170E', '#72190E')
#TolRainbow18 = (
#    '#D1BBD7', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987',
#    '#CAE0AB', '#F7F056', '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#A5170E', '#72190E')
#TolRainbow17 = (
#    '#D1BBD7', '#BA8DB4', '#AA6F9E', '#994F88', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987',
#    '#CAE0AB', '#F7F056', '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#72190E')
TolRainbow16 = (
    '#D1BBD7', '#BA8DB4', '#AA6F9E', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB',
    '#F7F056', '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#72190E')
#TolRainbow15 = (
#    '#D1BBD7', '#AA6F9E', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056',
#    '#F6C141', '#F1932D', '#E8601C', '#DC050C', '#72190E')
#TolRainbow14 = (
#    '#D1BBD7', '#AA6F9E', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056',
#    '#F6C141', '#F1932D', '#E8601C', '#DC050C')
#TolRainbow13 = (
#    '#D1BBD7', '#AA6F9E', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB', '#F7F056',
#    '#F4A736', '#E8601C', '#DC050C')
#TolRainbow12 = (
#    '#D1BBD7', '#AA6F9E', '#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#F4A736',
#    '#E8601C', '#DC050C')
#TolRainbow11 = ('#882E72', '#1965B0', '#5289C7', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#F4A736', '#E8601C', '#DC050C', '#72190E')
#TolRainbow10 = ('#882E72', '#1965B0', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#F4A736', '#E8601C', '#DC050C', '#72190E')
#TolRainbow9 = ('#882E72', '#1965B0', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#EE8026', '#DC050C', '#72190E')
#TolRainbow8 = ('#882E72', '#1965B0', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#EE8026', '#DC050C')
#TolRainbow7 = ('#882E72', '#1965B0', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#DC050C')
#TolRainbow6 = ('#1965B0', '#7BAFDE', '#4EB265', '#CAE0AB', '#F7F056', '#DC050C')
#TolRainbow5 = ('#1965B0', '#7BAFDE', '#4EB265', '#F7F056', '#DC050C')
#TolRainbow4 = ('#1965B0', '#4EB265', '#F7F056', '#DC050C')
#TolRainbow3 = ('#1965B0', '#F7F056', '#DC050C')
#
## Sam added
#TolRainbow2 = ('#1965B0', '#F7F056')
#TolRainbow1 = ('#1965B0',)

#ScatteredTolRainbow16 = tuple(np.array(TolRainbow16)[np.mod(np.arange(20)*5,20)])
ScatteredTurbo256 = tuple(np.array(bokeh_palettes.Turbo256)[np.mod(np.arange(256)*29,256)])
ScatteredCategory20_20 = bokeh_palettes.Category20_20[0::2]+bokeh_palettes.Category20_20[1::2]

def get_sam_palette(num_factors) -> tuple[str]:
    if num_factors<=10:
        return bokeh_palettes.Category10_10
    elif num_factors<=20:
        return ScatteredCategory20_20
    else:
        return ScatteredTurbo256
GnYlRd=bokeh_palettes.RdYlGn[11]
RdYlGn=tuple(reversed(bokeh_palettes.RdYlGn[11]))

Viridis256=bokeh_palettes.Viridis256
