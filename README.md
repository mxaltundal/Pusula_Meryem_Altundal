# Pusula_Meryem_Altundal
this repository made for the case work of pusula academy

Meryem Altundal 
meryem.6.altundal@gmail.com

Requirements
-Jupyter notebook
-python 3.0

You can download jupyter notebook & python 3.0 by this command
```
#Install the classic Jupyter Notebook with:
pip install notebook

#To run the notebook:
jupyter notebook
```


kurulum için gerekli paketler ilk hücrede mevcut

## Dosya içeriği:
kaynak dataset xlsx formatında
eda.jpynp eda ve preprocessing pipeline  notebook
prep_outputs içerisinde hedef feature ve diğer featurelar ayrılmış durumda fakat fold edilmiş halde değiller ayrı csv de bulunmaktalar
buna ek olarak ftr_ort özel bir subset oluşturuldu bunun nedeni pipeline notebook u içerisinde bulunmakta.

## dosya çalıştırma
run all dediğinizde sorunsuz çalışacaktır

## mevcut dosya içerisinde yapılanlar
-çoklu ve tekil kategorik datalar normalize edildi
-outlier detection ve sayısal veride na handle süreci tamamlandı
-veri hedef ve bağımlı featurelar olarak setlere ayırıldı
-alakalı özellikler tutuldu
-alerji cinsiyet uyruk verilerinde na handle tamamlandı

# Veri Hazırlama ve İnceleme Raporu

Proje: Talent Academy Case DT 2025
Hazırlayan: Meryem Altundal
Tarih: (güncel tarih)

## 0) Kapsam, Zamanlama ve Çalışma Süresi

Amaç: Veri setini model kurulumundan önce, tek başına kullanılabilir bir “ready-to-model” formata getirmek ve veri hakkında özet + teknik içgörü sunmak.

Model kurma yok: Bu çalışma model eğitimi yapmaz; veri hazırlığını tamamlayıp devreder.

Gecikme nedeni: Tez bitirme süreci ve şehir dışında olma nedeniyle planlanan tempoda ilerlenemedi.

Aktif çalışma süresi: Toplamda yaklaşık 4 gün odaklı çalışma ile tamamlandı.

Teslimatlar (çıktılar):

prep_outputs/XY_full.csv — Tüm veri, tüm özellikler (X) + hedef (y).

prep_outputs/XY_ftr_ort.csv — “FTR/Ortopedi” alt kümesi (KG_* etkisiz).

(Opsiyonel) prep_outputs/feature_schema.json — Özellik şeması/mapping (tutarlılık için önerilir).

## 1) Girdi, Ortam ve Yöntem

Kaynak: ~/Desktop/Pusula_Meryem_Altundal/data/Talent_Academy_Case_DT_2025.xlsx (ilk sayfa)

Kullanılan kütüphaneler: pandas, numpy, scikit-learn (KNNImputer, StandardScaler, NearestNeighbors).

Çalışma biçimi: Python betiği (Notebook’la uyumlu), adım adım log veren, index hizası için ekstra güvenlik kontrolleri içeren bir hat.

## 2) Adım Adım Pipeline — “Ne, Neden, Nasıl?”

Aşağıdaki adımlar sıra garantili çalışır; her satır düşüşünden sonra index reset yapılarak hizalama korunur. Bu, tipik “Unalignable boolean Series…” hatasını (boolean maske–index uyumsuzluğu) kökten engeller.

### 2.1 Kolon Adı Standardizasyonu

Ne: Tüm kolon adlarını küçük harfe indirip tüm boşlukları sildim.
Örn: "Uygulama Yerleri" → "uygulamayerleri", "Kan Grubu" → "kangrubu".

Neden: Kodun her yerinde tek tip isim kullanmak, yazım hatalarından kaçınmak, farklı dosyalardaki varyantları tek seferde normalize etmek.

Nasıl: df.columns = [re.sub(r"\s+","",str(c)).lower().strip() for c in df.columns].

### 2.2 Metin Normalizasyonu

Ne:

Tekil kategorikler (cinsiyet, kangrubu, bolum, tedaviadi, uyruk) → lower + trim + fazla boşlukları tekilleştirme.

Liste (multi-label) kolonları (kronikhastalik, alerji, tanilar, uygulamayerleri) → virgül biçimini a, b, c formatında sabitleme.

Neden: Aynı kavramın farklı yazımlarından doğan “sahte kategori”leri azaltmak.

Nasıl: Basit string fonksiyonları (norm_text, fix_commas), boş string → NaN dönüşümü.

### 2.3 Hedef ve Süre Türetilmesi

Ne:

tedavisuresi → içindeki ilk sayıyı alıp tedavisuresi_num olarak sayısallaştırdım.

uygulamasuresi → uygulamasuresi_num.

İkisi birden varsa toplamuygulamasuresi_dk = tedavisuresi_num * uygulamasuresi_num.

Neden: Model tarafında sayısal hedef ve yardımcı süreler gerek; metin biçimiyle çalışmak zor.

Nasıl: Regex ile ilk tamsayı yakalama → float.

### 2.4 Aykırı Değer (Outlier) Temizliği — Sadece yas

Ne: IQR (çeyrekler arası aralık) yöntemiyle uç yaş değerlerini düşürdüm.

Neden: Aşırı uçlar hem imputasyonu hem de ölçeklemeyi bozabilir.

Nasıl: Q1-1.5*IQR–Q3+1.5*IQR aralığı dışında kalanları filtreledim.

Kritik: Her satır düşüşünden sonra df.reset_index(drop=True).

### 2.5 Tekil Kategoriklerde Eksik Veri Politikası

Kangrubu — özel KNN imputasyonu

Ne: “bilinmiyor” yerine, benzer kayıtların kan grubunu atadım (sınıf imputasyonu).

Neden: Kan grubu klinik olarak anlamlı olabilir; mod ile doldurma yerine örnek-benzerliği kullanmak daha mantıklı.

Nasıl:

Özellik seti:

Sayısallar: yas, uygulamasuresi_num

Basit OHE: cinsiyet, bolum (en sık 15 + __rare__)

Liste uzunlukları: n_kronikhastalik, n_alerji, n_tanilar, n_uygulamayerleri

Bu özellikleri basitçe standardize ettim.

NearestNeighbors (k=7) ile en yakın komşuları buldum; 1/(mesafe+ε) ağırlıklı oyla sınıf belirledim.

Yedek kural: hiç anlamlı komşu yoksa global mod (en sık sınıf) kullanıldı.

Cinsiyet — eşikli politika

Ne: Eksik oranına göre kolon düş / satır düş / ‘bilinmiyor’.

Neden: Çok az eksik varsa satır düşmek daha güvenli; çok fazlaysa bilgi kaybı olmasın diye “bilinmiyor” olarak işaretlemek daha iyi.

Nasıl: MISS_COL_HIGH=0.35, MISS_ROW_LOW=0.05 eşikleriyle.

Uyruk — iş kuralı

Ne: Kolonu komple düşürdüm.

Neden: Analizde kullanılmayacağı net; gürültü ve veri yükü yapmasın.

### 2.6 Multi-label Eksik Doldurma

Ne:

alerji, kronikhastalik → NaN olan satıra açıkça ["yok"] token’ı verdim.

tanilar, uygulamayerleri → hedef (tedavisuresi_num) ile korelasyonu en yüksek token(lar) içerisinden top-k (k=1) ile boş kayıtları doldurdum.

Neden: Multi-label alanlarda “gerçekten yok” ile “bilinmiyor”u ayırmak önemli; “yok”u açıkça yazmak modelin anlamasına yardım eder. tanilar/uygulama yerleri tarafında ise hedefle ilişkili sinyali kaçırmamak için korelasyon ipucu kullanıldı.

Nasıl: Token → 0/1 vektör; Pearson korelasyonu; eşik |r| ≥ 0.15.

### 2.7 Sayısal Eksik Doldurma — KNNImputer

Ne: Sayısal kolonlarda (yas, uygulamasuresi_num vb.) KNNImputer (k=5).

Neden: Basit ortalama/medyan yerine, komşuluk bilgisi genelde daha isabetli doldurma verir.

Nasıl: StandardScaler ile ölçekle → KNNImputer → inverse transform.

Dikkat: Hedef (tedavisuresi_num) imputasyona dahil edilmez (sızıntı engeli).

### 2.8 Özellik Üretimi (Encoding)

Multi-label → Binarizasyon

Ne: Sık geçen token’lar için 0/1 sütunlar; ayrıca _count (kaç tane) ve _none (hiç yok mu) bayrakları.

Neden: Modelin nadir/gürültülü tokenlara boğulmaması; yeterli sıklığa odak.

Nasıl: RARE_THRESHOLD=10’un altındaki token’lar bırakıldı (binarizasyona girmedi).

Tekil kategorikler → OHE

cinsiyet → Cins_...

kangrubu → KG_...

bolum → Bolum_...

Yüksek kardinalite tedaviadi → rare bucket + OHE

Ne: Nadir tedavi adlarını __rare__ altında toplayıp OHE’ledim.

Neden: 200+ farklı tedavi adı, her biri için sütun açılırsa aşırı seyrek/gürültülü X oluşur.

Sızıntı önleme

Ne: tedavisuresi, tedavisuresi_num, toplamuygulamasuresi_dk X’ten çıkarıldı.

Neden: Hedefi veya onun deterministik türevini X’e koymak leakage yaratır.

### 2.9 Alt Küme — FTR/Ortopedi

Ne: bolum içinde ftr|fizik|fizyoterapi|ort|ortopedi geçen kayıtlar seçildi → XY_ftr_ort.csv.

Ek: Bu alt sette kan grubunun etkisini nötrlemek için KG_* sütunları düşürüldü.

Neden: Bölüm değişkeni hem çok dengesiz hem de çok güçlü; alt küme modelleyen ekipler için pratik bir başlangıç seti.

### 2.10 Kayıt ve Kontroller

Ne:

XY_full.csv ve XY_ftr_ort.csv yazıldı.

Kod sonunda boyut/iyi-kötü örnek sayıları log’landı.

Kritik kontrol: assert X_full.index.equals(df.index) — hizalama garantisi.

Neden: Reprodüksiyon ve güven; modelci ekip “hazır X, hazır y” ister.

3) Veri Dengesine Dair Gözlemler (Kısa Özet)

Cinsiyet: Kadın ~%57, Erkek ~%35, Bilinmiyor ~%8 → makul denge; “bilinmiyor” sinyali korunmalı.

Kan grubu: 0 Rh+ & A Rh+ toplu ~%73; küçük sınıflar %1–4 bandında → OHE’de gürültü riski, rare/gruplama ile kontrol.

Bölüm: FTR ~%93, ORT ~%4, diğerleri çok küçük → çok dengesiz ve hedefte büyük ayrım yaratıyor (kullanıcı ekibi bunu bilsin).

Alerji / Tanılar / Tedavi Adı: uzun kuyruk; yazım varyantları ve nadirler var → normalize/rare bucket ile kontrol edildi.

Not: Bu dağılım satırları, senin paylaştığın tabloların özüdür. Model kurulmayacağı için burada “ne yapılmalı” önerileri yalnızca bilgi amaçlı eklendi.

## 4) Sınırlar, Riskler ve Alınan Önlemler

Index hizası hataları: Her satır düşüşünden sonra reset_index(drop=True) → çarpışmaları engelledik.

Eksik veri: KNN impute (sayısal) + KNN sınıf imputasyonu (kangrubu) + net kurallar (yok, bilinmiyor).

Nadir kategoriler: RARE_THRESHOLD ile kontrollü; tedaviadi için rare bucket.

Leakage riski: Hedef ve türevleri X’ten çıkarıldı.

Yorumlanabilirlik: OHE/binarize sütun adları insan-okur; ayrıca şema JSON’u önerildi.

## 5) Model Kuracak Ekip İçin Kullanım Notları (Bilgi Amaçlı)

Değerlendirme: Bölüm dengesizliğini düşünerek branş bazlı metrik raporlayın (FTR vs ORT).

Düzenlileştirme: Uzun kuyruk + yüksek boyut için L2/L1/ElasticNet (lineer) ve erken durdurma/derinlik sınırı (ağaç) önerilir.

Kesit analizleri: Cinsiyet, kan grubu, bölüm gibi gruplara göre performans kontrolü yapın.

Tutarlılık: Eğitimde üretilen özellik şemasını (sütun ad/sıra) servis/inference tarafında bire bir koruyun.

## 6) Sonuç

Veri seti temiz, sayısallaştırılmış ve sızıntıdan arındırılmış halde teslim edildi.

Hem genel XY_full hem de FTR/ORT odaklı alt set mevcut.

Dengesizlikler, nadir kategoriler ve eksik veri stratejileri raporlandı.

Modelleme yapılmasa da, veri artık doğrudan herhangi bir eğitim hattına plug-and-play girer.

## Ek-1) Parametreler (varsayılanlar)
Parametre	Değer	Açıklama
RARE_THRESHOLD	10	Binarizasyon/OHE’de nadirlik eşiği
CORR_THRESH	0.15	tanilar/uygulamayerleri için korelasyon eşiği
TOPK_TOKENS	1	Korelasyona göre doldurulacak token sayısı
KNN_NEIGHBORS	5	Sayısal KNN imputasyonu
MISS_COL_HIGH	0.35	Tekil kategorikte çok eksikse kolon düş
MISS_ROW_LOW	0.05	Tekil kategorikte az eksikse satır düş
k (kangrubu KNN)	7	En yakın komşu sayısı (sınıf imputasyonu)

Bu değerler, veri büyüklüğüne ve dengesine göre ayarlanabilir.
