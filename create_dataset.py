import logging
from pathlib import Path
from itertools import islice

import h5py
import numpy as np
from simfile.dir import SimfilePack
from simfile.base import BaseSimfile as Simfile
from simfile.base import BaseChart as Chart
from simfile.notes import NoteData, NoteType
from simfile.notes.count import count_grouped_notes
from simfile.notes.group import group_notes, SameBeatNotes
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine


overcollection_folder: Path = Path('raw_data')
permissible_notetypes = {NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD, NoteType.TAIL}


def get_song(collection_name: str,
             pack_index: int = 1,
             song_index: int = 1) -> Simfile:
    overcollection: Path = overcollection_folder / collection_name
    folders = [SimfilePack(f) for f in overcollection.iterdir() if f.is_dir()]

    simfile_pack = folders[pack_index]
    song = next(islice(simfile_pack.simfiles(), song_index, None))
    return song

def iterate_notes_from_chart(chart: Chart):
   """Used for other stuff."""
   return group_notes(NoteData(chart),
                       include_note_types=permissible_notetypes,
                       same_beat_notes=SameBeatNotes.JOIN_ALL,
                       join_heads_to_tails=True)

def produce_tokens(chart: Chart, timing_engine: TimingEngine=None) -> tuple[float, float, int, int, int, int]:
  assert chart.stepstype == 'dance-single'
  notegroup_iterator: tuple[NoteData] = iterate_notes_from_chart(chart)
  for notegroup in notegroup_iterator:
    beat = notegroup[0].beat
    cols: list[int] = [n.column for n in notegroup]
    # col_arr: 4-tuple of bools
    col_arr = tuple(1 if (i in cols) else 0 for i in range(4))
    if timing_engine is not None:  # include the times
        time: float = timing_engine.time_at(beat)
        yield (beat, time, *col_arr)  # Beat, Decimal, 0/1, 0/1, 0/1, 0/1
    else:
       yield (beat, 0, *col_arr)

def process_simfile(simfile: Simfile) -> dict[str, list]:
    # We're gonna make one giant numpy array for each song with difficulties and all that.
    ds = dict()
    charts = [chart for chart in simfile.charts if chart.stepstype == 'dance-single']
    chart_sizes = [count_grouped_notes(iterate_notes_from_chart(c)) for c in charts]
    dtypes = [
      ('Beat', np.float16),
      ('Time', np.float16),
      ('c0', bool),
      ('c1', bool),
      ('c2', bool),
      ('c3', bool)]
   
    for i, chart in enumerate(charts):
        output = np.empty(chart_sizes[i], dtype=dtypes)
        timing_engine = TimingEngine(TimingData(simfile, chart))
        output[:] = list(produce_tokens(chart, timing_engine))
        ds[chart.meter] = output
    return ds

if __name__ == "__main__":
    """ Converts a nested directory of simfile packs into an hdf5 file
    which has beat, timing, and column data. """
    # TODO: Make an argparser.
    dataset_file = 'data.hdf5'
    # Delete if existing.
    Path(dataset_file).unlink(missing_ok=True)
    datasets = ['fraxtil', 'gpop', 'ddr']
    for ds in datasets:
        dataset_dir = overcollection_folder / ds
        folders = [SimfilePack(f) for f in dataset_dir.iterdir() if f.is_dir()]
        for folder in folders:
            simfile_pack = folder
            for song in simfile_pack.simfiles():
                h5path = f"/{dataset_dir.name}/{folder.name}/{song.title}"
                with h5py.File(dataset_file, 'a') as h5file:
                    h5file.require_group(h5path)
                    h5file[h5path].attrs['title'] = song.title
                songdata = process_simfile(song)
                logging.info(f"Saving {song.title} to {h5path}")
                with h5py.File(dataset_file, 'a') as h5file:
                    for k, v in songdata.items():
                        try:
                            h5file[f"{h5path}/{k}"] = v
                        except OSError:
                            pass  # Hmm.
                        h5file[f"{h5path}/{k}"].attrs['title'] = song.title
                        h5file[f"{h5path}/{k}"].attrs['difficulty'] = int(k)