import quandl
quandl.ApiConfig.api_key = 'puJtYkz3w2mjsUvx_38R'
data = quandl.get('HKEX/03988', column_index='1')
print(data)