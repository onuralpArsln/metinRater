# Metin Sınıflandırma Testleri Özeti (Workflow)

Bu döküman, projedeki 6 farklı Python test dosyasının metinleri analiz ederken tam olarak neye baktığını, kelimelerin sırasının/anlamının ne kadar önemli olduğunu ve sonuçların nasıl yorumlanacağını açıklamaktadır.

---

## Test 1: Basit Kelime Frekansı (Temel TF-IDF)
* **Odak Noktası:** Sadece kelimelerin **sıklığına (frekansına)** bakar. Kelimelerin sırasının hiçbir önemi yoktur. "Araba kırmızı" ile "Kırmızı araba" tamamen aynıdır.
* **Nasıl Çalışır:** Hangi kelimenin kaç defa geçtiğini sayar. Ancak İngilizce filtre (stop-words) kullandığı için Türkçe metinlerde hatalı sonuçlar verir.
* **Neye Bakmaz:** Anlama, kelime sırasına, büyük/küçük harfe, noktalama işaretlerine.
* **Sonuç Ne İfade Eder:** Ekrana basılan "Skor" (0 ile 1 arası), metnin eski başarılı veya başarısız metinlerle içerdiği **ortak kelime sayısının ve sıklığının** benzerlik yüzdesidir. Çıkan PCA grafiğindeki noktaların yakınlığı, kullanılan kelimelerin ne kadar aynı olduğunu gösterir.

## Test 2: Kelime ve İkili Kelime Dizilimleri (Gelişmiş TF-IDF)
* **Odak Noktası:** Hem tek tek kelimelerin sıklığına hem de **kısmi kelime sırasına** (yan yana gelen 2 kelimeye) bakar.
* **Nasıl Çalışır:** Türkçe bağlaçları temizler. Örneğin "kaliteli" kelimesini sayarken, aynı zamanda "çok kaliteli" ikilisinin de yan yana gelme **sıklığına** bakar.
* **Neye Bakmaz:** Cümlenin genel anlamı, büyük/küçük harf, noktalama.
* **Sonuç Ne İfade Eder:** Yeni test metninin, geçmişteki metinlerle ne kadar fazla ikili kelime kalıbı (örn: "boyun destekli") paylaştığını gösterir. Yüksek skor, geçmişte başarı getiren kelime kalıplarını birebir kopyaladığınız anlamına gelir.

## Test 3: Yapay Zeka ile Kelime Ağırlığı Puanlaması (Lojistik Regresyon)
* **Odak Noktası:** Kelimelerin frekansı önemlidir, kelime sırası (yan yana 2 kelime olarak) önemlidir. Anlam önemli değildir.
* **Nasıl Çalışır:** TF-IDF ile kelimeleri sayar (Test 2'deki gibi) ama benzerlik ölçmek yerine bir "Yapay Zeka" (Machine Learning) modeli eğitir. Modele kelimeleri gösterir ve "Hangi kelimenin geçmesi başarı şansını ne kadar artırıyor?" sorusunun cevabını hesaplatır.
* **Sonuç Ne İfade Eder:** Sonuçlar sadece benzerlik değil, **Net Matematiksel Olasılıktır (Confidence %)**. Eğer %95 Başarılı diyorsa, model içinde geçen kelimelerin (örn: "ergonomik" kelimesinin +2.5 puanlık ağırlığının) matematiksel olarak başarmaya yeterli olduğuna karar vermiştir. Bar grafiği (Feature Importance), hangi kelimelerin en büyük "torpil" sağladığını açıkça gösterir.

## Test 4: Derin Yapay Zeka ile Anlamsal Analiz (Semantik NLP)
* **Odak Noktası:** Sadece cümlenin **GENEL ANLAMINA ve BAĞLAMINA** bakar. Kelime frekansı, sırası (anlamı değiştirmiyorsa) ve tam eşleşme hiç önemli değildir.
* **Nasıl Çalışır:** Dünya çapında eğitilmiş çok dilli bir dil modeli (Sentence Transformer) kullanır. "Harika bir ürün" ile "Çok beğendim, kalite tesadüf değildir" cümlelerinde ortak bir kelime olmasa bile bu cümlelerin aynı "hissiyatı ve anlamı" barındırdığını bilir.
* **Neye Bakmaz:** Hangi kelimenin kaç defa geçtiğine (kelime sayımı yapmaz).
* **Sonuç Ne İfade Eder:** Olasılık veya kelime skoru değil, **Anlamsal Mesafe (Semantic Similarity)** verir. Yüksek bir skor, yeni yazılan metnin, geçmişteki başarılı metinlerin yaydığı "duygu veya mesaja" çok benzediği anlamına gelir. (En zeki ve insan gibi düşünen test budur.)

## Test 5: Harf ve Hece Taraması (Karakter N-Gram)
* **Odak Noktası:** Sadece harf parçalarının (3 ila 5 harflik grupların) **sıklığına** ve harf büyüklüğüne bakar. Kelime sırası veya anlam umrunda değildir. "Kelimeler" bile umrunda değildir.
* **Nasıl Çalışır:** "HARİKA!!!" yazıldığında sırasıyla "HAR", "ARİ", "RİK", "İKA", "KA!", "A!!", "!!!" gruplarının geçme sıklığını hesaplar.
* **Neye Bakmaz:** Metnin diline, anlamına, mantığına. 
* **Sonuç Ne İfade Eder:** Bir metnin biçimsel (format) olarak geçmiştekilere ne kadar benzediğini gösterir. Sonuçlar (Olasılık Yüzdesi), modelin aşırı büyük harf, ardışık noktalama işaretleri veya benzer ekler (örneğin -iyorlar eki) içeren metin kalıplarını yakaladığını gösterir. Çoğunlukla "Spam" özellikli sahte mesajları yakalamakta kullanılır.

## Test 6: Noktalama Merkezli Sınıflandırma (Özel Tokenizasyon)
* **Odak Noktası:** Tam kelimelerin ve özel olarak **noktalama işareti dizilerinin frekansına** bakar. Kelime sırası önemli değildir.
* **Nasıl Çalışır:** "Büyük" ile "büyük" kelimesini farklı sayar. "???" grubunu, "ürün" kelimesi gibi tek başına anlamlı bir kelime olarak sınıflandırır ve sayar.
* **Sonuç Ne İfade Eder:** Olasılık Skorları (Confidence %), yazarın tarzını (kamera arkası klavye alışkanlıklarını) puanlamaktır. Eğer başarısız metinler genelde bol ünlem (!) veya virgüller (,) ile yazılıyorsa, yeni test metni çok virgüle sahip olduğu için "Başarısız" veya "Başarılı" kategorisine daha fazla çekilecektir. Sonuç, noktalama kalıplarının kelimeler kadar güçlü bir kader belirleyici olduğu anlamına gelir.
