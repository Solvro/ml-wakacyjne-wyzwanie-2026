Jeśli są jakieś niejasności można się skontaktować na discord: `domin._` lub `mccobblerola`

# Google Colab

W czasie wakacyjnego kursu będziemy głównie operować na notatnikach Jupyter na platformie Google Colab (jeśli nie masz jeszcze konta na googlu to pora założyć). Więc warto byłoby się zapoznać jak taki notatnik przygotować:

## Stworzenie notatnika

Wchodząc na strone: [https://colab.research.google.com](https://colab.research.google.com/) powinniśmy zostać przywitani taką stroną:
[![alt text](<data/Zrzut ekranu 2026-07-27 o 22.58.53.png>)](<data/Zrzut ekranu 2026-07-27 o 22.58.53.png>)
Jeśli pierwszy raz masz do czynienia z Google Colab warto zapoznać się z wyświetlanym poradnikiem. Ale nas interesuje stworzenie nowego notatnika, kliknij `New notebook`. W ten oto sposób utworzyłeś nowy notatnik, na którym będziesz mógł pracować.

> Jeśli z jakiegoś powodu nie pokazało Ci się okno `Open Notebook`, nowy notebook możesz zrobić poprzez kliknięcie `File -> New Notebook` in Drive.

## Używanie notatnika

Po stworzeniu notatnika, Google Colab stworzy nam segment kodu na którym możemy testować nasz kod i odpalać program. 

[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.08.40.png>)](<data/Zrzut ekranu 2026-07-27 o 23.08.40.png>)

Dalszy poradnik na przykładowym kodzie z prostym modelem uczenia maszynowego (regresji liniowej).
[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.23.33.png>)](<data/Zrzut ekranu 2026-07-27 o 23.23.33.png>)

> Możesz zauwazyć, że ostatnia linijka w segmencie kodu, działa tak jak print 🔥

Zeby dodać kolejne segmenty kodu wystarczy najechać myszką pod segment kodu i pojawi nam się:
[![alt text](<data/Zrzut ekranu 2026-07-28 o 09.34.18.png>)](<data/Zrzut ekranu 2026-07-28 o 09.34.18.png>)

Można także dodawać komentarze markdown, klikając `+ Text`,  które są bardzo przydatne przy opisywaniu co się w danym segmencie kodu dzieje:
[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.16.48.png>)](<data/Zrzut ekranu 2026-07-27 o 23.16.48.png>)

Dużą zaletą takiego notatnika, jest to że odpalony wcześniej segment kodu działa wobec innych segmentów kodu (zachowuje kontekst), więc możemy testować kod fragmentami:
[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.18.54.png>)](<data/Zrzut ekranu 2026-07-27 o 23.18.54.png>)

zmienne X i Y, które zostały zdefiniowane wcześniej działają dla nowego segmentu!

> **⚠️ Środowisko Colaba jest ulotne**
>
> Twój notatnik działa na tymczasowej maszynie wirtualnej i łatwo stracić stan pracy. Są dwa poziomy „resetu":
>
> - **Restart sesji** (`Runtime → Restart session`) restartuje tylko kernel Pythona — **znikają wszystkie zmienne** (stan w pamięci), ale pliki na dysku i doinstalowane biblioteki zostają.
> - **Rozłączenie lub usunięcie środowiska** (bezczynność, przekroczenie limitu czasu, `Factory reset`) tworzy nową maszynę — **znika wszystko**: zmienne, biblioteki z `!pip install` i pliki zapisane w `/content`.
>
> Sesja rozłącza się po ok. 90 min bezczynności i ma maksymalny czas życia ok. 12 h (darmowy plan; Google tych limitów nie gwarantuje i bywają zmieniane).
>
> **Wniosek:** wszystko, co chcesz zachować — dane wejściowe i wyniki — trzymaj na Dysku Google (`/content/drive/MyDrive/...`). Dysk jest podmontowany z zewnątrz, więc przeżywa restart maszyny. To, co leży tylko w `/content`, traktuj jak brudnopis.

## Importowanie bibliotek w notatniku

Nie raz będziemy potrzebowali zaimportować jakąś libke w notatniku (Google Colab ma już wgrane w środowisko dużo przydatnych bibliotek), ale tutaj przecież nie ma żadnego terminala, żeby użyć pip install, w notatniku jupytera robimy to wewnątrz segmentu kodu:

[![alt text](<data/Screenshot 2025-07-26 at 01.56.11.png>)](<data/Screenshot 2025-07-26 at 01.56.11.png>)

W notatniku Jupyter:

- wykrzyknik umożliwia wykonanie polecenia systemowego (shellowego) z poziomu komórki notebooka (działa jak wpisanie komendy w terminalu)
- to są tak zwane "magic commands", czyli specjalne polecenia Jupyter Notebooka, które dają dodatkowe (jeśli jesteś ciekawy co oferuje Jupyter Notebook możesz wpisać w kodzie `%lsmagic`, żeby wyprintować wszystkie metody).

## Ustawienia notatnika

Wchodząc w Edit -> Notebook Settings, możecie między innymi przestawić wykonywanie programu z CPU na GPU, jeśli chcecie i także schować podpowiadanie kodu przez AI

[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.21.34.png>)](<data/Zrzut ekranu 2026-07-27 o 23.21.34.png>)

## Ładowanie stworzonego juz notatnika

Jeśli wchodzicie znowu na strone Google Colab i chcecie dokończyć robote co zostawiliście sobie na później, na ekranie startowym dostaniecie automatycznie skąd chcecie odpalić poprzedni notebook

[![alt text](<data/Zrzut ekranu 2026-07-27 o 23.23.10.png>)](<data/Zrzut ekranu 2026-07-27 o 23.23.10.png>)

a jak chcecie się przełączyć z jednego notebooka na drugi to poprzez `File -> Open notebook` i dostaniecie to samo okno.

I tak - wyniki (`print`, wykresy) są zapisywane wewnątrz pliku `.ipynb`, więc pobrany notatnik zachowuje je gotowe do wglądu dla nas — pod warunkiem że przed pobraniem uruchomisz notatnik od nowa (**Restart session and run all**), żeby zapisane wyniki odpowiadały aktualnemu kodowi.

---

# GitHub

Cały proces będzie oddawania zadanek wyglądał następująco:

- **Wasz branch `dev`:** W głównym repozytorium kursu wygenerujemy dla każdego z was osobnego brancha `user-dev`. To na niego będziecie wysyłać wszystkie swoje rozwiązania przypisanych zadań.

    [![alt text](<data/Zrzut ekranu 2026-07-28 o 00.23.22 1.png>)](<data/Zrzut ekranu 2026-07-28 o 00.23.22 1.png>)
    [![alt text](<data/Zrzut ekranu 2026-07-28 o 00.23.33.png>)](<data/Zrzut ekranu 2026-07-28 o 00.23.33.png>)

- **Zgłaszanie rozwiązania (Pull Request):** Gdy zadanie będzie gotowe, waszym kolejnym krokiem będzie utworzenie Pull Requesta (PR) z waszego brancha na główny branch `main`. Będzie to stanowić oficjalne przekazanie kodu do sprawdzenia.
- **Oznaczenie autora (Assignee):** Jedyne, o czym musicie pamiętać przy tworzeniu PR-a, to przypisanie do niego siebie w polu "Assignee". Dzięki temu wiemy, do kogo należy dane rozwiązanie.
- **Review i zamknięcie zadań:** My zajmujemy się resztą procesu. Przydzielimy do waszych PR odpowiednich reviewerów, a po upływie ustalonego terminu i zakończeniu oceniania zamkniemy wasze zgłoszenia.

## Kroki do wykonania:

### 1. Pobierz notebook z maina z GitHuba

1. Wejdź na repozytorium na GitHubie.
2. Przejdź do branch `main`.
3. Znajdź interesujący Cię plik notebooka (`.ipynb`).

    [![alt text](<data/Zrzut ekranu 2026-07-28 o 09.36.09.png>)](<data/Zrzut ekranu 2026-07-28 o 09.36.09.png>)

4. Skopiuj adres URL z paska przeglądarki.

### 2. Importuj notebook w Google Colab

1. Otwórz [Google Colab](https://colab.research.google.com/).
2. Wybierz **File** > **Open notebook**.
3. Przejdź do zakładki **GitHub**.
4. Wklej wcześniej skopiowany URL pliku notebooka z GitHuba.

    [![alt text](<data/Screenshot 2025-07-29 at 18.24.48.png>)](<data/Screenshot 2025-07-29 at 18.24.48.png>)
    [![alt text](<data/Zrzut ekranu 2026-07-27 o 23.48.54.png>)](<data/Zrzut ekranu 2026-07-27 o 23.48.54.png>)

5. Kliknij **Search** (Szukaj), a następnie wybierz swój plik i kliknij na niego. Wtedy powinien otworzyć Ci się notebook

_Alternatywnie_: możesz wybrać **File > Open notebook > Upload** i załadować notebook pobrany z komputera.

### 3. Podepnij Dysk Google w Colabie

1. W Colabie wklej poniższy kod w nową komórkę i uruchom ją:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Daj potrzebne permisje google colabowi, po daniu permisji:

    [![alt text](<data/Zrzut ekranu 2026-07-27 o 23.59.02.png>)](<data/Zrzut ekranu 2026-07-27 o 23.59.02.png>)

Po tym powinieneś móc pod tą ścieżką wylistować wszystkie swoje pliki w dysku `%ls /content/drive/MyDrive`

### 4. Wgraj dane na swój Dysk Google

1. Wejdź na [Google Drive](https://drive.google.com/).
2. Przeciągnij plik/dane do odpowiedniego folderu na Dysku.
3. Następnie w notebooku możesz odwołać się do tego pliku np.:

    ```python
    data_path = '/content/drive/MyDrive/nazwa_folderu/nazwa_pliku.csv'
    ```

### 5. Pracuj w notatniku, dokonaj potrzebnych zmian

- Wykonaj analizy/kod zgodnie z zadaniem.

### 6. Pobierz skończony notatnik

1. W Colabie wybierz **File (Plik) > Download > Download .ipynb** (Pobierz jako .ipynb).

[![alt text](<data/Zrzut ekranu 2026-07-28 o 00.07.22.png>)](<data/Zrzut ekranu 2026-07-28 o 00.07.22.png>)

2. Plik zostanie zapisany na Twoim komputerze.

### 7. Wgraj notatnik na SWOJEGO brancha `user-dev` w repozytorium GitHub

Zachowaj **tę samą nazwę pliku** co oryginalny notebook — inaczej upload tworzy duplikat zamiast czystego diffa.

[![alt text](<data/Zrzut ekranu 2026-07-28 o 00.23.22.png>)](<data/Zrzut ekranu 2026-07-28 o 00.23.22.png>)

1. Kliknij **Add file > Upload files** i wybierz pobrany plik notebooka.

[![alt text](<data/Zrzut ekranu 2026-07-28 o 00.09.32.png>)](<data/Zrzut ekranu 2026-07-28 o 00.09.32.png>)

2. Zatwierdź zmiany (commit).

### 8. Stwórz Pull Request

1. Po wgraniu zmian pojawi się propozycja utworzenia Pull Requesta.
2. Kliknij **Compare & pull request**.

    [![alt text](<data/Zrzut ekranu 2026-07-28 o 00.11.56.png>)](<data/Zrzut ekranu 2026-07-28 o 00.11.56.png>)

3. Uzupełnij opis, upewnij się, że porównywane są właściwe gałęzie, zmiany powinny zostać wprowadzone na `main` (`user-dev` → `main`).

    [![alt text](<data/Zrzut ekranu 2026-07-28 o 00.13.34 1.png>)](<data/Zrzut ekranu 2026-07-28 o 00.13.34 1.png>)

4. Kliknij **Create pull request**.

## Pull requesty

Tutaj przedstawimy prostą mechanike jak można stworzyć pull request na dowolnym repozytorium.

Załóżmy, że mamy takie o to repozytorium:

[![alt text](<data/Screenshot 2025-07-26 at 18.26.19.png>)](<data/Screenshot 2025-07-26 at 18.26.19.png>)

Wchodząc w zakładke pull requests klikamy `New Pull Request` i przejdziemy na stronę, gdzie możemy porównać naszego brancha z jakimś innym (na tym przykładzie zrobimy pull request na main):

[![alt text](<data/Screenshot 2025-07-26 at 18.29.57.png>)](<data/Screenshot 2025-07-26 at 18.29.57.png>)

> Base przy porównywaniu to jest branch, w którym chcemy wprowadzić zmiany (`main`) a compare to branch, z którego pochodzą zmiany (`dev`).

I następnie klikamy `Create pull request`

[![alt text](<data/Screenshot 2025-07-26 at 18.32.54.png>)](<data/Screenshot 2025-07-26 at 18.32.54.png>)

Przy pull requeście możemy wybrać kto będzie robił recenzje danych zmian i możemy także przypisać osobę odpowiedzialną za wykonanie (klikając zębatke przy reviewers i assignee odpowiednio). Kiedy stworzymy już adekwatny tytuł i opis oraz przypiszemy osoby do pull requestu klikamy `Create pull request`. (cross-review)

Kiedy już stoi taki pull request czekamy od osób trzeci na adekwatny roast zmian/nowych feature'ów.

Teraz, w jaki sposób możemy jakiejś osobie zrobić recenzje? W naszym repozytorium w zakładce Pull Requests, możemy zauważyć, wszystkie aktywne pull requesty na repozytorium:

[![alt text](<data/Screenshot 2025-07-26 at 18.39.22.png>)](<data/Screenshot 2025-07-26 at 18.39.22.png>)

Wchodząc w interesujący nas pull request, mamy opcje:

- Napisanie prostego komentarza:

[![alt text](<data/Screenshot 2025-07-26 at 18.41.06.png>)](<data/Screenshot 2025-07-26 at 18.41.06.png>)

Ale także komentarze, które odnoszą się do jakiegoś kawałka kodu, możemy to zrobić poprzez wejście w zakładke przy pull requeście `Files Changed` i klikając interesującą nas linijke:

[![alt text](<data/Screenshot 2025-07-26 at 18.45.52.png>)](<data/Screenshot 2025-07-26 at 18.45.52.png>)

i kliknąć `Start a review`. Wtedy na głównej stronie pull requestu pojawi się:

[![alt text](<data/Screenshot 2025-07-26 at 18.46.56.png>)](<data/Screenshot 2025-07-26 at 18.46.56.png>)

Jako iż w pull request jupyter notebooki są pokazywane jakby w formacie jsona, dobrym pomysłem jest przejście z pull requesta na faktyczny notebook i tam na niego spojrzeć i w pull request napisać normalny komentarzy (nie koniecznie zaznaczając fragment jsona).
[![alt text](<data/Zrzut ekranu 2026-07-28 o 00.20.28.png>)](<data/Zrzut ekranu 2026-07-28 o 00.20.28.png>)

Ostatecznie jeśli wszystko zostało rozwiązane czy to jakiś konflikt czy jakieś zrequestowane zmiany w kodzie osoba która robi approve i klika `Merge Pull Request` i w ten oto sposób zmiany z brancha zostały wrzucone na main.
[![alt text](<data/Zrzut ekranu 2026-07-28 o 00.21.33.png>)](<data/Zrzut ekranu 2026-07-28 o 00.21.33.png>)

## Dodatek: Commity i podstawowe komendy Gita

Małe wtrącenie: w tym konkretnym zadaniu nie będziecie musieli robić commitów bezpośrednio z poziomu kodu w Colabie. Opieramy się na pracy z całym plikiem – wystarczy, że na koniec po prostu wgracie gotowy, pobrany notatnik na waszego brancha przez stronę Githuba. Będzie to wasz jedyny, całościowy commit.

Ale jako że **commit to integralna część Gita**, na pewno przyda wam się ta wiedza w przyszłości, żeby rozumieć, jak to działa. Mówiąc najprościej: commit to po prostu taki "zapis stanu gry".

Żeby zapisać i wysłać swoje postępy na bieżąco (np. gdy pracujecie nad większym projektem), używa się standardowo trzech komend. W Colabie wpisujecie je w komórce z kodem, dodając na początku wykrzyknik:

- `!git add .` ➔ To polecenie zbiera wszystkie zmienione pliki. Mówicie Gitowi: "przygotuj te nowości do zapisu". Kropka na końcu oznacza, że dodajemy wszystkie bieżące zmiany.
- `!git commit -m "krótki opis zmian"` ➔ To jest właściwy "zapis stanu gry". Zamykacie przygotowane zmiany w paczkę i opisujecie co zostało zmienione.
- `!git push` ➔ Ta komenda bierze waszą gotową paczkę (commita) i wysyła ją do waszego repozytorium, na waszego brancha na githubie.

_(Dla ułatwienia, Colab pozwala to też po prostu wyklikać: wchodzicie w górne menu **File** ➔ **Save a copy in GitHub**, wpisujecie opis zmian, a platforma sama wykonuje powyższe kroki w tle)._

## Discord
