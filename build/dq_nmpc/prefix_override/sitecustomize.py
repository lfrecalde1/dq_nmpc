import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/ext3/ws_acp/src/dq_nmpc/install/dq_nmpc'
