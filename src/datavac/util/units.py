from pint import UnitRegistry

from datavac.util.logging import logger

units = UnitRegistry()

class Normalizer():
    def __init__(self, deets):
        self._udeets={}
        self._shorthands={}
        for n,(shorthand,ninfo) in deets.items():
            self._shorthands[n]=shorthand
            self._udeets[n]={}
            if '[' in n:
                norm_units=units.parse_expression(n.split('[')[1].split(']')[0])
            else:
                norm_units=units.parse_units('1')
            for ks,kinfo in ninfo.items():
                if type(ks) is not tuple:
                    ks=(ks,)
                for k in ks:
                    self._udeets[n][k]=kinfo.copy()
                    if '[' in k:
                        start_units_from_name=k.split('[')[1].split(']')[0]
                        if ('start_units' in self._udeets[n][k]) \
                                and self._udeets[n][k]['start_units']!=start_units_from_name:
                            raise Exception(f"Under '{n}', column '{k}'" \
                                            f"has conflicting start units {self._udeets[n][k]['start_units']}")
                        else:
                            self._udeets[n][k]['start_units']=start_units_from_name
                    else:
                        if 'start_units' not in self._udeets[n][k]:
                            raise Exception(f"Under '{n}', no start_units provided or read from column '{k}'")
                    start_units=units.parse_expression(self._udeets[n][k]['start_units'])
                    end_units=units.parse_units(self._udeets[n][k]['end_units'])
                    assert (ntype:=self._udeets[n][k]['type']) in ['*', '/']
                    nstart_units=start_units/norm_units if ntype=='/' else start_units*norm_units
                    try:
                        self._udeets[n][k]['units_scale_factor']=nstart_units.to(end_units).magnitude
                    except Exception as e:
                        logger.error(f"Couldn't convert '{k}' in {start_units} with normalization '{n}' to {end_units}")
                        raise e

    def get_scaled(self, df, column, normalizer):
        if normalizer is False: return df[column]
        if column not in self._udeets[normalizer]:
            logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return df[column]
        ntype=self._udeets[normalizer][column]['type']
        scale=self._udeets[normalizer][column]['units_scale_factor']
        column=df[column]
        if hasattr(column,'to_numpy'):
            column=column.to_numpy()
        normalizer=df[normalizer] if normalizer!='None' else 1
        if hasattr(normalizer,'to_numpy'):
            normalizer=normalizer.to_numpy()
        res= (column/normalizer if ntype=='/' else column*normalizer)*scale
        if hasattr(res,'to_numpy'):
            res=res.to_numpy()
        return res

    def shorthand(self, column, normalizer):
        if (normalizer is False) or (column not in self._udeets[normalizer]):
            #logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return ""
        t={'/':'/','*':r'\cdot '}[self._udeets[normalizer][column]['type']]
        sh=self._shorthands[normalizer]
        return f"{t}{sh}" if sh!="" else ""

    def formatted_endunits(self, column, normalizer):
        if (normalizer is False) or (column not in self._udeets[normalizer]):
            #logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return ""
        eu=self._udeets[normalizer][column]['end_units']\
            .replace("*",r'$\cdot$').replace("ohm",r"$\Omega$")\
            .replace("u",r"$\mu$").replace("$$","")\
            .replace("^2",r"$^2$")
        return eu

    def normalizer_columns(self):
        return [k for k in self._udeets if k != 'None']

    @property
    def norm_options(self):
        return list(self._udeets.keys())


