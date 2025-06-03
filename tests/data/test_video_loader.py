

def test_video_iterator(video_file):

    from beast.data.video import VideoFrameIterator

    batch_size = 8
    iterator = VideoFrameIterator(video_file=video_file, batch_size=batch_size)
    batch = next(iterator)
    assert batch['image'].shape == (batch_size, 3, 224, 224)
    assert 'video' in batch
    assert 'idx' in batch
    assert 'image_path' in batch
    del iterator

    batch_size = 4
    iterator = VideoFrameIterator(video_file=video_file, batch_size=batch_size)
    batch = next(iterator)
    assert batch['image'].shape == (batch_size, 3, 224, 224)
    del iterator

    # make sure last batch is handled fine
    n_frames = 0
    batch_size = 32
    iterator = VideoFrameIterator(video_file=video_file, batch_size=batch_size)
    for batch in iterator:
        n_frames += batch['image'].shape[0]
    assert n_frames == iterator.total_frames
    del iterator
