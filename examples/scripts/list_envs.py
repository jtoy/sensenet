from sensenet import envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
