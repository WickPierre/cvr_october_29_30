from ultralytics import YOLO


model = YOLO('/Users/petrlutkin/Desktop/cvr_october_29_30/runs/detect/train/weights/best.pt')


results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/831b783d-56e0-5509-b00f-3b9e6c0600e6.jpeg',
    conf=0.2,
    save=True
)

results = model.predict(
    source='/Users/petrlutkin/Desktop/cvr_october_29_30/6450209278.jpg',
    conf=0.2,
    save=True
)