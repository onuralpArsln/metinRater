# Metin Sınıflandırma Testleri Özeti (Workflow)

Bu döküman, projedeki 6 farklı Python test dosyasının metinleri analiz ederken tam olarak neye baktığını ve nasıl çalıştığını basit bir dille açıklamaktadır.

## Test 1: Basit Kelime Eşleştirme (Temel TF-IDF)
* **Neye Bakar:** Sadece tek tek kelimelere bakar.
* **Nasıl Çalışır:** Kelimeleri sayar ve metnin geneline oranlar. Ancak İngilizce filtre (stop-words) kullandığı için Türkçedeki "ve", "ama", "bir" gibi bağlaçları elemeyi başaramaz.
* **Kullanım Amacı:** Sadece en ilkel ve hızlı temel bir karşılaştırma yapmak içindir, Türkçe için yetersizdir.

## Test 2: Gelişmiş Kelime ve Kelime Grubu Eşleştirme (Gelişmiş TF-IDF)
* **Neye Bakar:** Hem tekil kelimelere (örn: "boyun") hem de ikili kelime gruplarına (örn: "boyun destekli") yanyana bakar.
* **Nasıl Çalışır:** Türkçe dil filtresi kullandığı için gereksiz kelimeleri ("ve", "ile") başarıyla temizler. Geçmişteki başarılı metinlerin "ortalama kelime profili" ile yeni metni karşılaştırır.
* **Kullanım Amacı:** Birebir aynı kelimelerin veya kelime öbeklerinin geçtiği metinleri yakalamak için idealdir.

## Test 3: Yapay Zeka ile Puanlama (Lojistik Regresyon)
* **Neye Bakar:** Hangi kelimenin başarıya, hangi kelimenin başarısızlığa neden olduğuna karar verir.
* **Nasıl Çalışır:** Test 2'deki kelime gruplarını alır ama ortalama almak yerine bir "Makine Öğrenmesi" modeli eğitir. "Ergonomik" kelimesinin başarı şansını %20 artırdığını, "Battaniye" kelimesinin %15 düşürdüğünü matematiksel olarak hesaplar.
* **Kullanım Amacı:** Hangi kelimelerin metni daha başarılı yaptığını net bir puanla (olasılık yüzdesiyle) görmek istediğinizde kullanılır.

## Test 4: Anlamsal Analiz (Semantik Derin Öğrenme)
* **Neye Bakar:** Kelimelerin yazılışına değil, cümlenin **toplam anlamına ve bağlamına** bakar.
* **Nasıl Çalışır:** Çok dilli gelişmiş bir yapay zeka modeli (Sentence Configuration) kullanır. "Harika bir ürün" ile "Çok beğendim, kalite tesadüf değildir" cümlelerinde hiç ortak kelime olmasa bile, aynı anlama geldiklerini anlayabilir. Noktalama işaretlerinin yarattığı hissiyatı bile kavrar.
* **Kullanım Amacı:** İnsan gibi okuyup, metinlerin ana fikrine ve anlamına göre gruplandırma yapmak istediğinizde kullanılır (En zeki modeldir).

## Test 5: Harf ve Hece Analizi (Karakter N-Gram)
* **Neye Bakar:** Kelimelerin tamamına değil, 3 ila 5 harflik parçalara (hecelere veya uzun harf dizilerine) bakar. 
* **Nasıl Çalışır:** Büyük/küçük harf ayrımı yapar ("A" ile "a" aynı değildir). Örneğin "MÜK" harf dizisini veya "!!!" üçlü ünlem dizisini yakalar. Kelimelerin son eklerini veya köklerini otomatik olarak parçalamış olur.
* **Kullanım Amacı:** Hepsi büyük harfle yazılmış bağıran metinleri, tekrarlayan harfleri ("Süppppper") veya benzer ekleri alan farklı kelimeleri yakalamak için kullanılır.

## Test 6: Noktalama ve Özel Karakter Analizi (Özel Tokenizasyon)
* **Neye Bakar:** Tam kelimelere ve yan yana gelmiş noktalama işaretlerine (bağımsız birer kelimeymiş gibi) bakar.
* **Nasıl Çalışır:** Büyük harfleri korur. Cümledeki "!!!!", "???", ",", "|" gibi karakter gruplarını kelimelerden ayırarak onlara özel bir önem (ağırlık) verir.
* **Kullanım Amacı:** Kullanıcıların noktalama işareti kullanım alışkanlıklarını, örneğin aşırı ünlem kullananların veya virgül ile düzgün cümle kuranların başarı/başarısızlık durumunu ölçmek için idealdir. 
