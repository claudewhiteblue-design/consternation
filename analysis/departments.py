# -*- coding: utf-8 -*-
"""Department classification shared by the overview (131) and the analyses engine (132).

NONFOOD is what the "food & drink only" sample drops. Alcohol is NOT here -- it is a
drink, priced per litre like any other. "אביזרים לארוח" is disposable tableware and is
already out of every regression through EXCAT; it is listed so the classification reads
complete. Two departments that look like equipment are food and stay in: "עזרי אפייה
ובישול" is entirely flour, and "מזנונים" is entirely sandwiches.
"""
NONFOOD=['אביזרים ומוצרי תינוקות','תכשירי כביסה','מוצרי נייר','היגיינה וטיפוח הגוף',
         'ניקוי הבית','אביזרים לארוח','היגיינת הפה','טיפוח השיער','סבוני רחצה',
         'שטיפת כלים','מוצרי גילוח','מוצרי שיזוף והגנה מהשמש','טיפוח פנים']
