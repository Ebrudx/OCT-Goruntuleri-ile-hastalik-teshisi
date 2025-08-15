readme_content = """
#  OCT Tomografi Göz Hastalığı Tahmin Sistemi

Bu proje, **EfficientNetB0** tabanlı derin öğrenme modeli kullanarak retina OCT (Optical Coherence Tomography) görüntülerinden göz hastalıklarını otomatik olarak tespit eder.  
Gradio arayüzü sayesinde kullanıcılar OCT görüntüsünü yükleyerek hastalık adını, kısa açıklamasını, çözüm önerilerini ve tahmin doğruluk oranını görebilir.


##  Proje Amacı
Retina OCT görüntüleri üzerinden **AMD, CNV, CSR, DME, DR, DRUSEN, MH** ve **NORMAL** sınıflarını tespit ederek göz hastalıklarının teşhis sürecine destek olmak.  
Bu sistem yalnızca **destekleyici** niteliktedir, kesin tanı için göz uzmanına başvurulmalıdır.

---

##  Kullanılan Veri Seti
- **Kaynak:** [Retinal OCT C8 Dataset - Kaggle](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)
- **Sınıflar:**
  - AMD
  - CNV
  - CSR
  - DME
  - DR
  - DRUSEN
  - MH
  - NORMAL



##  Kurulum

1. **Depoyu klonlayın**
```bash
git clone https://github.com/kullanici_adiniz/oct-tomografi-tahmin.git
cd oct-tomografi-tahmin
