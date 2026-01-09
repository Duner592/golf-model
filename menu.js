   function toggleMenu() {
       const menu = document.getElementById('menu');
       const overlay = document.getElementById('overlay');
       const body = document.body;
       menu.classList.toggle('show');
       overlay.classList.toggle('show');
       body.classList.toggle('menu-open');
   }

   function toggleSection(header) {
       const list = header.nextElementSibling;
       const arrow = header.querySelector('.arrow');
       if (list.style.display === 'block') {
           list.style.display = 'none';
           arrow.textContent = '▶';
       } else {
           list.style.display = 'block';
           arrow.textContent = '▼';
       }
   }
