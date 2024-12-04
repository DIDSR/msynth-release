

$.fn.DataTable.ext.pager.numbers_length = 10;

$(document).ready(function () {
    $.getJSON("metadata.json").done(function (data) {
        $.each(data["columns"], function(i,e){
            $("#data-table thead tr").append(`<th>${e}</th>`);
            $("#data-table tfoot tr").append(`<th>${e}</th>`);
        });
        $.each(data["references"], function(i,e){
            $("#references").append(`<a href="${e["url"]}">${e["title"]}</a> `);
        });
        $(".title").text(data["title"]);
        $("title").text(data["title"]+" DEMO");
        $.each(data["images"], function (idx, e) {
            tr=$('#data-table').append(`<tr id="data-${idx}">`);
            $.each(e,function(idx2,e2){
                if(new RegExp(["jpg","png","gif"].join("|")).test(e2)){
                    $(`#data-${idx}`).append(`<td><img src="images/${e2}" style="width:100px" class="sample"></td>`);
                }else{
                    $(`#data-${idx}`).append(`<td>${e2}</td>`);
                }
            });
        });

        $.get("README.md").done(function (data) {
            $(".content[content='description']").html(marked.parse(data));
            $(".content[content='description']").show();
        });
        
        let table = new DataTable('#data-table',
            {
                order: [[0, 'asc']],
                layout: {
                    
                    topStart: 'info',
                    bottom: 'paging',
                    bottomStart: null,
                    bottomEnd: null
                },                
                initComplete: function () {
                    this.api()
                        .columns()
                        .every(function () {
                            let column = this;

                            if(column.type()=="html-num"){
                                column.footer().replaceChildren("");
                                return;
                            }
                            let title = column.footer().textContent;
             
                            // Create input element
                            let input = document.createElement('input');
                            input.className="footer-search";
                            input.placeholder = title;
                            column.footer().replaceChildren(input);
             
                            // Event listener for user input
                            input.addEventListener('keyup', () => {
                                if (column.search() !== this.value) {
                                    column.search(input.value).draw();
                                }
                            });
                        });
                }
            }
        );
        Fancybox.bind('img', {
            //
          }); 
    });

    $(".tabs li").click(function(){
        $(".tabs li").removeClass("active");
        $(this).addClass("active");
        tab=$(this).attr("content");
        console.log(tab);
        $(".content").hide();
        $(".content[content='"+tab+"']").show();
        DataTable.tables({ visible: true, api: true }).columns.adjust();
    });
});