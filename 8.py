from ultralytics import YOLO


model = YOLO('/Users/petrlutkin/Desktop/cvr_october_29_30/runs/detect/train/weights/best.pt')


results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/00105.jpg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/02924.jpg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/02833.jpg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/02975.jpg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/02631.jpg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/archive/test_dataset/test_images/00074.jpg',
    conf=0.2,
    save=True
)