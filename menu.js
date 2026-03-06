   function toggleMenu(force) {
       const menu = document.getElementById('menu');
       const overlay = document.getElementById('overlay');
       const body = document.body;
       if (!menu || !overlay) {
           return;
       }
       const shouldOpen = force === undefined ? !menu.classList.contains('show') : force;
       if (shouldOpen) {
           menu.classList.add('show');
           overlay.classList.add('show');
           body.classList.add('menu-open');
       } else {
           menu.classList.remove('show');
           overlay.classList.remove('show');
           body.classList.remove('menu-open');
       }
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

   document.addEventListener('click', event => {
       const target = event.target;
       if (target instanceof Element) {
           const link = target.closest('.menu-link');
           if (link) {
               toggleMenu(false);
           }
       }
   });
