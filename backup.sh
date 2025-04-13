datetime=$(date +%Y%m%d_%H%M%S)
mkdir -p snapshots/$datetime
cp *.db *.shelve snapshots/$datetime