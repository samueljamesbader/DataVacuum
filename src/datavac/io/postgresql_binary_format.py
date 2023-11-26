# See under Binary Format
# https://www.postgresql.org/docs/16/sql-copy.html
import io
import struct
from functools import partial

import pandas as pd
from sqlalchemy import Column

pd_to_pg_converters= {
    'INT64': lambda x: int(x).to_bytes(length=4,signed=True),
    'INT32': lambda x: int(x).to_bytes(length=4,signed=True),
    'FLOAT64': lambda x: struct.pack("!d", float(x)),
    'FLOAT32': lambda x: struct.pack("!d", float(x)),
    'STRING': partial(str.encode,encoding='utf-8'),
    'BOOLEAN': partial(bool.to_bytes,length=1),
}
pg_to_pd_converters= {
    'INTEGER': partial(int.from_bytes,signed=True),
    'DOUBLE_PRECISION': partial(struct.unpack,"!d"),
    'VARCHAR': partial(bytes.decode,encoding='utf-8'),
    'BOOLEAN': bool.from_bytes,
}

def df_to_pgbin(df: pd.DataFrame, override_converters={}):
    converters=[(override_converters.get(c,None) or pd_to_pg_converters[str(dtype).upper()])\
                for c,dtype in df.dtypes.items()]
    return data_to_pgbin(df.to_records(index=False),converters)

def data_to_pgbin(data,converters):
    bio=io.BytesIO()

    # Header signature
    bio.write(b'PGCOPY\n\377\r\n\0')
    # Header flags
    bio.write(b'\0\0\0\0')
    # No header extension
    bio.write(b'\0\0\0\0')

    for row in data:

        # Write the number of fields in the row
        num_fields=len(row)
        bio.write(num_fields.to_bytes(2,signed=True))

        # For each field
        for field,converter in zip(row,converters):

            # Try to convert to binary
            try:
                cfield=converter(field)

            # If it fails, write a NULL
            except Exception as e:
                #print(e)
                bio.write(b'\xff\xff\xff\xff')

            # Or if it succeeds, write the size, then the field
            else:
                size=len(cfield)
                bio.write(size.to_bytes(4,signed=True))
                bio.write(cfield)

    # End marker
    bio.write(b'\xff\xff')

    bio.seek(0)
    return bio

def pgbin_to_df(bio: io.BytesIO, tab_columns: list[Column], override_converters={}):
    converters=[override_converters[c.name] if c.name in override_converters
                    else pg_to_pd_converters[c.type.__class__.__name__]
                for c in tab_columns]
    recs=pgbin_to_data(bio,converters)
    return pd.DataFrame(recs,columns=[c.name for c in tab_columns])


def pgbin_to_data(bio: io.BytesIO,converters):
    bio.seek(0)
    data=[]

    # Header signature check
    assert bio.read(11)==b'PGCOPY\n\377\r\n\0'
    # Flags check
    assert bio.read(4)==b'\0\0\0\0'
    # No header extension check
    assert bio.read(4)==b'\0\0\0\0'

    while True:
        row=[]

        num_fields=int.from_bytes(num_field_bytes:=bio.read(2))
        if num_field_bytes==b'\xff\xff': break
        assert num_fields==len(converters), f"Why are there {num_fields} fields in this row?"

        for _,converter in zip(range(num_fields),converters):
            size=int.from_bytes(bio.read(4),signed=True)
            if size==-1:
                row.append(None)
            else:
                field=bio.read(size)
                assert len(field)==size, "Insufficient data left"
                row.append(converter(field))
        data.append(row)
    return data
