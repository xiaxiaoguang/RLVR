"""Freeze final prelaunch handoff files without rewriting the runtime/smoke receipt."""
from common import *

def main():
    assert not (RUN/'main.claimed').exists()
    assert read(RUN/'verification.json')['status']=='passed'
    assert read(RUN/'signal-tests.json')['status']=='passed'
    validate_sources(delivery=True)
    result={}
    for p in sorted(RUN.rglob('*')):
        rel=str(p.relative_to(RUN))
        if rel.startswith('tmp/') or rel=='deliverable-identities.json' or rel.endswith('.writing'):continue
        if p.is_symlink():result[rel]={'symlink':os.readlink(p)}
        elif p.is_file():result[rel]=content(p)
    result['shared_handoff']={'path':'/work/users/lab_admin/shared/IDEA-20261121-01/HANDOFF.md',**content('/work/users/lab_admin/shared/IDEA-20261121-01/HANDOFF.md')}
    write(RUN/'deliverable-identities.json',result)
    print('Final prelaunch handoff frozen',len(result),'entries; runtime receipt unchanged')

if __name__=='__main__':main()
